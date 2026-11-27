"""
Cell-qualified heuristic default lookup workflow (Section 3.11).

Input: condition pair, CFG scale, model family.
Output: recommended weights or "no recommendation".

The tool returns prespecified default weights ONLY for the Canny+Depth CFG 7.5 cell
(38:62 Canny:Depth). For all other cells (CFG15, unmeasured model families,
OpenPose/Scribble ratio recommendations), it returns "no recommendation",
even if SII cells are measured, because no ratio-level preference evaluation
is available. SII is NEVER used to set or rank weights.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union


@dataclass
class LookupResult:
    """Result from the default weight lookup tool."""
    recommendation: Optional[Dict[str, float]]
    message: str
    cell_measured: bool
    ratio_evaluated: bool


class DefaultLookupTool:
    """
    Cell-qualified heuristic default lookup.
    
    Rules:
    - Returns prespecified 38:62 (Canny:Depth) ONLY for (canny_depth, 7.5, sdxl).
    - Returns "no recommendation" for all other cells.
    - SII is never used to set or rank weights.
    - "Auto" refers to automatic querying of the measured lookup table,
      not automatic optimization.
    """

    # Measured SII cells (have slope measurements but NOT ratio-level evaluation)
    _MEASURED_CELLS = {
        ("canny_depth", 7.5, "sdxl"),
        ("canny_depth", 15.0, "sdxl"),  # CFG15 forward only
        ("openpose_canny", 7.5, "sdxl"),
        ("canny_openpose", 7.5, "sdxl"),
        ("scribble_depth", 7.5, "sdxl"),
        ("depth_scribble", 7.5, "sdxl"),
    }

    # Cells with ratio-level preference evaluation (usability-tested)
    _RATIO_EVALUATED_CELLS = {
        ("canny_depth", 7.5, "sdxl"),
    }

    # Prespecified heuristic defaults (NOT derived from SII, NOT measured optima)
    _DEFAULTS = {
        ("canny_depth", 7.5, "sdxl"): {
            "canny": 0.38,
            "depth": 0.62,
        },
    }

    def __init__(self):
        pass

    def lookup(
        self,
        pair: str,
        cfg: float,
        model: str = "sdxl",
    ) -> LookupResult:
        """
        Query the lookup table.
        
        Parameters
        ----------
        pair : str
            Condition pair identifier, e.g. "canny_depth", "openpose_canny", "scribble_depth".
        cfg : float
            CFG scale, e.g. 7.5 or 15.0.
        model : str
            Model family: "sdxl", "sd15", "flux".
        
        Returns
        -------
        LookupResult with recommendation or "no recommendation".
        """
        key = (pair, cfg, model)

        cell_measured = key in self._MEASURED_CELLS
        ratio_evaluated = key in self._RATIO_EVALUATED_CELLS

        if ratio_evaluated and key in self._DEFAULTS:
            return LookupResult(
                recommendation=self._DEFAULTS[key],
                message=(
                    f"Prespecified heuristic default for {pair} at CFG {cfg} on {model}: "
                    f"{list(self._DEFAULTS[key].keys())[0]}:{self._DEFAULTS[key][list(self._DEFAULTS[key].keys())[0]]:.2f}, "
                    f"{list(self._DEFAULTS[key].keys())[1]}:{self._DEFAULTS[key][list(self._DEFAULTS[key].keys())[1]]:.2f}. "
                    "This is a prespecified heuristic default, not a measured optimum, "
                    "not derived from SII."
                ),
                cell_measured=True,
                ratio_evaluated=True,
            )

        if cell_measured:
            return LookupResult(
                recommendation=None,
                message=(
                    f"Cell ({pair}, CFG {cfg}, {model}) has SII slope measurements, "
                    "but no ratio-level preference evaluation is available. "
                    "No recommendation returned. Do not use SII to set or rank weights."
                ),
                cell_measured=True,
                ratio_evaluated=False,
            )

        return LookupResult(
            recommendation=None,
            message=(
                f"Cell ({pair}, CFG {cfg}, {model}) has not been measured. "
                "No recommendation returned. Extrapolation beyond measured conditions "
                "is not performed."
            ),
            cell_measured=False,
            ratio_evaluated=False,
        )

    def get_sii_identifiability_matrix(self) -> Dict[str, bool]:
        """
        Return SII identifiability status for model families.
        SDXL CFG7.5 selected cells are measured (SII identifiable).
        SD1.5 and Flux use normalized-ratio designs without fixed-A single-B runs,
        so SII is not identifiable by design.
        """
        return {
            "sdxl_cfg75_canny_depth_forward": True,
            "sdxl_cfg75_canny_depth_reverse": True,
            "sdxl_cfg75_openpose_canny_both": True,
            "sdxl_cfg75_scribble_depth_both": True,
            "sdxl_cfg15_canny_depth_forward": True,
            "sd15_all": False,  # normalized-ratio design, no fixed-A
            "flux_all": False,  # normalized-ratio design, no fixed-A
        }

    def __repr__(self) -> str:
        return (
            "DefaultLookupTool("
            "presets={canny_depth@CFG7.5/sdxl: canny:0.38 depth:0.62}, "
            "all_other_cells: no_recommendation)"
        )


def interactive_cli():
    """Command-line interface for the lookup tool."""
    print("=== Cell-Qualified Heuristic Default Lookup Tool ===")
    print("Enter condition pair (canny_depth / openpose_canny / scribble_depth),")
    print("CFG scale (7.5 / 15.0), and model (sdxl / sd15 / flux).")
    print("Type 'quit' to exit.\n")

    tool = DefaultLookupTool()

    while True:
        pair = input("Condition pair: ").strip().lower()
        if pair == "quit":
            break
        cfg_str = input("CFG scale: ").strip()
        model = input("Model (sdxl/sd15/flux): ").strip().lower() or "sdxl"

        try:
            cfg = float(cfg_str)
        except ValueError:
            print("Invalid CFG scale. Please enter a number.\n")
            continue

        result = tool.lookup(pair=pair, cfg=cfg, model=model)
        print(f"\n{result.message}\n")
        if result.recommendation:
            for cond, w in result.recommendation.items():
                print(f"  {cond}: {w:.2f}")
        print()


if __name__ == "__main__":
    interactive_cli()
