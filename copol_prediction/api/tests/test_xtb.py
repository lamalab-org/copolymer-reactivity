#!/usr/bin/env python3
"""
Test script to verify XTB and feature calculation dependencies work correctly.
Run this inside the Docker container to ensure all dependencies are properly installed.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path to import morfeus_patch
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_imports():
    """Test basic Python imports."""
    print("=" * 60)
    print("TEST 1: Basic Python imports")
    print("=" * 60)
    
    try:
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ numpy: FAILED - {e}")
        return False
    
    try:
        import pandas
        print(f"✓ pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"✗ pandas: FAILED - {e}")
        return False
    
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print(f"✓ rdkit: {rdkit_version}")
    except ImportError as e:
        print(f"✗ rdkit: FAILED - {e}")
        return False
    
    print("\n")
    return True


def test_morfeus():
    """Test morfeus import and basic functionality."""
    print("=" * 60)
    print("TEST 2: Morfeus (quantum chemistry library)")
    print("=" * 60)
    
    try:
        from morfeus.conformer import ConformerEnsemble
        print("✓ morfeus.conformer imported")
    except ImportError as e:
        print(f"✗ morfeus.conformer: FAILED - {e}")
        return False
    
    try:
        # Try to import patched XTB class first
        try:
            from morfeus_patch import XTB
            print("✓ morfeus.XTB imported (using patched version)")
        except ImportError:
            from morfeus import XTB
            print("✓ morfeus.XTB imported (using original version)")
    except ImportError as e:
        print(f"✗ morfeus.XTB: FAILED - {e}")
        return False
    
    print("\n")
    return True


def test_qcengine():
    """Test QCEngine (quantum chemistry engine interface)."""
    print("=" * 60)
    print("TEST 3: QCEngine")
    print("=" * 60)
    
    try:
        import qcengine
        print(f"✓ qcengine: {qcengine.__version__}")
    except ImportError as e:
        print(f"✗ qcengine: FAILED - {e}")
        return False
    
    try:
        import qcelemental
        print(f"✓ qcelemental: {qcelemental.__version__}")
    except ImportError as e:
        print(f"✗ qcelemental: FAILED - {e}")
        return False
    
    # Configure QCEngine to find xtb via environment variable
    try:
        import os
        xtb_path = "/opt/xtb/bin/xtb"
        if os.path.exists(xtb_path):
            # QCEngine looks for QC_{PROGRAM}_EXE environment variable
            os.environ["QC_XTB_EXE"] = xtb_path
            print(f"✓ QCEngine configured with XTB via QC_XTB_EXE: {xtb_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not configure QCEngine XTB path: {e}")
    
    print("\n")
    return True


def test_xtb_binary():
    """Test if XTB binary is available and executable."""
    print("=" * 60)
    print("TEST 4: XTB Binary")
    print("=" * 60)
    
    import os
    import subprocess
    
    # Check environment variables
    xtb_home = os.environ.get('XTBHOME', 'Not set')
    print(f"XTBHOME: {xtb_home}")
    
    # Check if xtb is in PATH
    try:
        result = subprocess.run(['which', 'xtb'], capture_output=True, text=True)
        if result.returncode == 0:
            xtb_path = result.stdout.strip()
            print(f"✓ XTB binary found: {xtb_path}")
        else:
            print("✗ XTB binary NOT found in PATH")
            return False
    except Exception as e:
        print(f"✗ Error checking XTB binary: {e}")
        return False
    
    # Try running xtb --version
    try:
        result = subprocess.run(['xtb', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_info = result.stdout.strip().split('\n')[0]
            print(f"✓ XTB version: {version_info}")
        else:
            print(f"✗ XTB execution failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ XTB execution timed out")
        return False
    except Exception as e:
        print(f"✗ Error running XTB: {e}")
        return False
    
    print("\n")
    return True


def test_simple_conformer():
    """Test simple molecule conformer generation (no XTB calculation yet)."""
    print("=" * 60)
    print("TEST 5: Simple Conformer Generation")
    print("=" * 60)
    
    try:
        from morfeus.conformer import ConformerEnsemble
        
        # Simple molecule: ethylene
        smiles = "C=C"
        print(f"Testing with SMILES: {smiles}")
        
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        print(f"✓ ConformerEnsemble created with {len(ce.conformers)} conformer(s)")
        
        ce.prune_rmsd()
        print(f"✓ RMSD pruning successful, {len(ce.conformers)} conformer(s) remaining")
        
        ce.sort()
        print("✓ Conformers sorted by energy")
        
    except Exception as e:
        print(f"✗ Conformer generation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n")
    return True


def test_xtb_calculation():
    """Test actual XTB quantum chemistry calculation."""
    print("=" * 60)
    print("TEST 6: XTB Quantum Chemistry Calculation")
    print("=" * 60)
    print("(This may take 30-60 seconds...)")
    print()
    
    try:
        from morfeus.conformer import ConformerEnsemble
        # Try to import patched XTB class first
        try:
            from morfeus_patch import XTB
        except ImportError:
            from morfeus import XTB
        
        # Very simple molecule for fast testing: ethylene
        smiles = "C=C"
        print(f"Testing with SMILES: {smiles}")
        
        # Generate conformer
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()
        print(f"✓ Conformer generated: {len(ce.conformers)} conformer(s)")
        
        # Try GFN-FF optimization
        print("  Attempting GFN-FF optimization...")
        try:
            ce.optimize_qc_engine(
                program="xtb", 
                model={"method": "GFN-FF"}, 
                procedure="geometric"
            )
            print("  ✓ GFN-FF optimization successful")
        except Exception as e:
            print(f"  ⚠ GFN-FF optimization failed (trying fallback): {e}")
            ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
            ce.prune_rmsd()
            ce.sort()
        
        # Try GFN2-xTB optimization
        print("  Attempting GFN2-xTB optimization...")
        try:
            ce.optimize_qc_engine(
                program="xtb", 
                model={"method": "GFN2-xTB"}, 
                procedure="geometric"
            )
            print("  ✓ GFN2-xTB optimization successful")
        except Exception as e:
            print(f"  ✗ GFN2-xTB optimization failed: {e}")
            traceback.print_exc()
            return False
        
        # Single point calculation
        print("  Attempting single point calculation...")
        try:
            ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
            print("  ✓ Single point calculation successful")
        except Exception as e:
            print(f"  ✗ Single point calculation failed: {e}")
            traceback.print_exc()
            return False
        
        # Get best conformer
        best_conformer = ce.conformers[0]
        print(f"✓ Best conformer energy: {best_conformer.energy:.6f} Hartree")
        
        # Calculate properties with XTB
        print("  Calculating molecular properties...")
        elements = best_conformer.elements.tolist()
        coordinates = best_conformer.coordinates.tolist()
        
        xtb = XTB(elements, coordinates)
        
        # Calculate key properties
        homo = xtb.get_homo()
        lumo = xtb.get_lumo()
        fukui_radical = xtb.get_fukui("radical")
        
        print(f"  ✓ HOMO: {homo:.6f}")
        print(f"  ✓ LUMO: {lumo:.6f}")
        print(f"  ✓ Fukui radical indices calculated: {len(fukui_radical)} atoms")
        
        if fukui_radical:
            fukui_max = max(fukui_radical.values())
            print(f"  ✓ Fukui radical max: {fukui_max:.6f}")
        
        print("\n✓ Full XTB calculation pipeline successful!")
        
    except Exception as e:
        print(f"\n✗ XTB calculation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  XTB Feature Calculation Dependency Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = []
    
    # Run all tests
    results.append(("Basic imports", test_basic_imports()))
    results.append(("Morfeus", test_morfeus()))
    results.append(("QCEngine", test_qcengine()))
    results.append(("XTB binary", test_xtb_binary()))
    results.append(("Conformer generation", test_simple_conformer()))
    results.append(("XTB calculation", test_xtb_calculation()))
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! XTB feature calculation is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

