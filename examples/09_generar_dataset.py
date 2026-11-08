"""Genera un dataset etiquetado (fractura sí/no + máscara) para entrenar una red neuronal.

Ejemplos:
    python examples/09_generar_dataset.py --modalidad ct --n 200
    python examples/09_generar_dataset.py --modalidad mri --n 200 --secuencia STIR
    python examples/09_generar_dataset.py --modalidad ultrasound --n 20     (≈1 min por imagen)

Carga en PyTorch, por ejemplo:
    d = np.load("outputs/dataset_ct/ct_00000.npz"); img, label, mask = d["image"], d["label"], d["mask"]
"""

import argparse
from pathlib import Path

from wavesim.datasets import MODALITIES, generate_dataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--modalidad", choices=MODALITIES, default="ct")
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--preset", default="tibia", choices=["tibia", "antebrazo", "muslo"])
parser.add_argument("--semilla", type=int, default=0)
parser.add_argument("--prob-fractura", type=float, default=0.5)
parser.add_argument("--secuencia", default="STIR", help="secuencia de RM (T1, T2, PD_FS, STIR, GRE, UTE)")
parser.add_argument("--salida", default=None)
args = parser.parse_args()

extra = {"sequence": args.secuencia} if args.modalidad == "mri" else {}
out = Path(args.salida) if args.salida else Path(__file__).resolve().parent.parent / "outputs" / f"dataset_{args.modalidad}"
generate_dataset(out, args.n, args.modalidad, args.preset, seed=args.semilla, fracture_prob=args.prob_fractura, **extra)
print("Dataset en", out)
