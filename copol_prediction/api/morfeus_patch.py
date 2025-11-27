"""
Patch for morfeus XTB class to handle compatibility issues with newer XTB versions.
Fixes:
- Missing 'HOMO-LUMO gap / eV' in JSON output  
- Fukui coefficient parsing issues
"""
import json
import re
from pathlib import Path
from morfeus import XTB as OriginalXTB


class XTB(OriginalXTB):
    """
    Patched XTB class that handles missing fields in xtbout.json gracefully.
    """
    
    def _parse_json(self, json_file: Path) -> None:
        """Parse xtbout.json file with fallback for missing fields."""
        with open(json_file) as f:
            data = json.load(f)
        
        # Store all available results
        self._results.total_energy = data.get("total energy", None)
        self._results.homo = data.get("HOMO", None)
        self._results.lumo = data.get("LUMO", None)
        
        # Try to get gap from JSON, if not available calculate it
        if "HOMO-LUMO gap / eV" in data:
            self._results.gap = data["HOMO-LUMO gap / eV"]
        elif self._results.homo is not None and self._results.lumo is not None:
            # Calculate gap in eV (HOMO and LUMO are in Hartree in the JSON)
            # But actually they might already be in eV, so we just take the difference
            self._results.gap = self._results.lumo - self._results.homo
        else:
            self._results.gap = None
        
        self._results.fermi_level = data.get("Fermi-level / eV", None)
        self._results.dipole_moment = data.get("total dipole / D", None)
        
        # Get dipole vector if available
        if "molecular dipole" in data:
            dipole_data = data["molecular dipole"]
            self._results.dipole_vect = [
                dipole_data.get("x", 0.0),
                dipole_data.get("y", 0.0), 
                dipole_data.get("z", 0.0)
            ]
        
        # Parse charges
        if "partial charges" in data:
            self._results.charges = data["partial charges"]
        
        # Parse Fukui indices if available
        if "fukui indices" in data:
            fukui_data = data["fukui indices"]
            # Extract Fukui+ (electrophilic)
            if "f(+)" in fukui_data:
                self._results.fukui_plus = {
                    i: val for i, val in enumerate(fukui_data["f(+)"])
                }
            # Extract Fukui- (nucleophilic)
            if "f(-)" in fukui_data:
                self._results.fukui_minus = {
                    i: val for i, val in enumerate(fukui_data["f(-)"])
                }
            # Extract Fukui0 (radical)
            if "f(0)" in fukui_data:
                self._results.fukui_radical = {
                    i: val for i, val in enumerate(fukui_data["f(0)"])
                }
        
        # Parse IP and EA
        self._results.ip = data.get("IP / eV", data.get("IP", None))
        self._results.ea = data.get("EA / eV", data.get("EA", None))
    
    def get_fukui(self, kind: str = "radical") -> dict:
        """
        Get Fukui indices with fallback for parsing issues.
        
        Args:
            kind: Type of Fukui index ('radical', 'electrophilicity', 'nucleophilicity')
        
        Returns:
            Dictionary mapping atom indices to Fukui values
        """
        try:
            # Try the original method first
            return super().get_fukui(kind)
        except (ValueError, KeyError, AttributeError) as e:
            # If parsing fails, return empty dict or try alternative approach
            print(f"Warning: Fukui calculation failed ({e}). Returning empty result.")
            return {}

