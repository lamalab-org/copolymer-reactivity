
feature_columns = [
        # Molecular descriptors for Monomer 1
        'best_conformer_energy_1', 'ip_1', 'ip_corrected_1', 'ea_1', 'homo_1', 'lumo_1',
        'global_electrophilicity_1', 'global_nucleophilicity_1', 'charges_min_1', 'charges_max_1',
        'charges_mean_1', 'fukui_electrophilicity_min_1', 'fukui_electrophilicity_max_1',
        'fukui_electrophilicity_mean_1', 'fukui_nucleophilicity_min_1', 'fukui_nucleophilicity_max_1',
        'fukui_nucleophilicity_mean_1', 'fukui_radical_min_1', 'fukui_radical_max_1',
        'fukui_radical_mean_1', 'dipole_x_1', 'dipole_y_1', 'dipole_z_1',

        # Molecular descriptors for Monomer 2
        'best_conformer_energy_2', 'ip_2', 'ip_corrected_2', 'ea_2', 'homo_2', 'lumo_2',
        'global_electrophilicity_2', 'global_nucleophilicity_2', 'charges_min_2', 'charges_max_2',
        'charges_mean_2', 'fukui_electrophilicity_min_2', 'fukui_electrophilicity_max_2',
        'fukui_electrophilicity_mean_2', 'fukui_nucleophilicity_min_2', 'fukui_nucleophilicity_max_2',
        'fukui_nucleophilicity_mean_2', 'fukui_radical_min_2', 'fukui_radical_max_2',
        'fukui_radical_mean_2', 'dipole_x_2', 'dipole_y_2', 'dipole_z_2',

        # HOMO-LUMO differences
        'delta_HOMO_LUMO_AA', 'delta_HOMO_LUMO_AB', 'delta_HOMO_LUMO_BB', 'delta_HOMO_LUMO_BA',

        # Other features
        'temperature', 'solvent_logp',
        'polytype_emb_1', 'polytype_emb_2', 'method_emb_1', 'method_emb_2'
    ]
