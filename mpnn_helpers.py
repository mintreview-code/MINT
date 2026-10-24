"""LF MPNN helpers extracted from the original greedy adapter; no immune-oracle imports."""

import numpy as np
import torch
from vendor.mpnn.data_utils import parse_PDB, featurize
from vendor.mpnn.model_utils import ProteinMPNN

RESTYPE_STRtoINT = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
ALPHABET = list(RESTYPE_STRtoINT)


def build_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        device=device,
        atom_context_num=1,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=0,
    )
    drop = ["features.side_chain_atom_types", "features.periodic_table_features"]
    checkpoint["model_state_dict"] = {
        k: v for k, v in checkpoint["model_state_dict"].items() if k not in drop
    }
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def setup_protein(pdb_path, fixed_residues, hlas, device):
    """Parse PDB, build the designable chain_mask, and featurize. Returns
    (feature_dict, backbone_prody, other_atoms, designable_global_idx, chain_filter)."""
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        pdb_path, device=device, atom_context_num=1, chains="", parse_all_atoms=False
    )

    R_idx = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters = list(protein_dict["chain_letters"])
    encoded = [f"{chain_letters[i]}{R_idx[i]}{icodes[i]}" for i in range(len(R_idx))]

    fixed_set = set(fixed_residues)
    fixed_positions = torch.tensor(
        [int(r not in fixed_set) for r in encoded], device=device
    )  # 1 where NOT fixed
    # design every chain; the immune windows never cross chains (single chain here)
    chain_mask = torch.ones(len(encoded), dtype=torch.int32, device=device)
    protein_dict["chain_mask"] = chain_mask * fixed_positions  # 1=design, 0=fixed
    protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

    protein_dict["membrane_per_residue_labels"] = torch.zeros_like(chain_mask)
    protein_dict["pssm"] = torch.zeros([len(encoded), 20], device=device)
    protein_dict["lambda_esm"] = 0.0
    protein_dict["checkpoint_esm"] = ""
    protein_dict["lambda_sequence_bias"] = 1.0
    protein_dict["immuno_HLAs"] = hlas
    protein_dict["kappa_deimmunise"] = 1.0

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=1,
        number_of_ligand_atoms=1,
        model_type="protein_mpnn",
    )

    designable = [
        i for i in range(len(encoded)) if protein_dict["chain_mask"][i].item() == 1
    ]
    return feature_dict, backbone, other_atoms, designable, encoded


def base_feature_common(feature_dict, temperature, omit_aa, device):
    L = feature_dict["S"].shape[1]
    omit_vec = torch.tensor(
        [1.0 if aa in set(omit_aa) else 0.0 for aa in ALPHABET],
        device=device,
        dtype=torch.float32,
    )
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = temperature
    feature_dict["bias"] = (-1e8 * omit_vec[None, None, :]).repeat([1, L, 1])
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    feature_dict["restype_STRtoINT"] = RESTYPE_STRtoINT
    feature_dict["score_immunogen_exp"] = "piecewiselinear"
    feature_dict["n_redesign_rounds"] = 1
    feature_dict["min_deimmune_pep_len"] = 9
    feature_dict["max_deimmune_pep_len"] = 11
    feature_dict["immuno_oracles"] = ["NetMHCpan"]
    feature_dict["Immuno_Oracle_class"] = None  # kappa=1 -> no soft guidance
    feature_dict["homo_oligomer"] = 0
    feature_dict["symmetry_sum_prob_space"] = False
    feature_dict["capsnetmhc_path"] = ""
    feature_dict["netmhcpan_path"] = ""
    feature_dict["lambda_sequence_bias"] = 1.0  # 1 == no bias toward input seq
    return feature_dict, omit_vec
