from dataclasses import dataclass
from pathlib import Path

from src.symbols import TARGET_LOGICAL_SYMBOLS


DIRECTIONS = ("LONG", "SHORT")


@dataclass(frozen=True)
class DirectionModelState:
    asset: str
    direction: str
    model_path: Path
    available: bool


@dataclass(frozen=True)
class AssetPipelineState:
    asset: str
    data_path: Path
    indicators_path: Path
    features_path: Path
    long_model: DirectionModelState
    short_model: DirectionModelState
    backtest_ready: bool
    walk_forward_ready: bool
    oof_ready: bool
    costs_ready: bool
    broker_aware_sizing_ready: bool
    frozen_model_ready: bool
    forward_test_ready: bool

    @property
    def models_available(self):
        return self.long_model.available and self.short_model.available

    @property
    def validated(self):
        return all(
            (
                self.models_available,
                self.backtest_ready,
                self.walk_forward_ready,
                self.oof_ready,
                self.costs_ready,
                self.broker_aware_sizing_ready,
                self.frozen_model_ready,
                self.forward_test_ready,
            )
        )

    @property
    def missing_validation(self):
        missing = []
        if not self.long_model.available:
            missing.append("modelo LONG")
        if not self.short_model.available:
            missing.append("modelo SHORT")
        if not self.backtest_ready:
            missing.append("backtest")
        if not self.walk_forward_ready:
            missing.append("walk-forward")
        if not self.oof_ready:
            missing.append("OOF")
        if not self.costs_ready:
            missing.append("costes")
        if not self.broker_aware_sizing_ready:
            missing.append("broker-aware sizing")
        if not self.frozen_model_ready:
            missing.append("modelo congelado")
        if not self.forward_test_ready:
            missing.append("forward test")
        return missing


def _has_costs_summary(path):
    if not path.exists():
        return False
    try:
        header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except IndexError:
        return False
    return "COST_BPS" in header and "THRESHOLD" in header


def _has_file(path):
    return path.exists() and path.is_file()


def _direction_state(raiz, asset, direction):
    path = raiz / "models" / asset.lower() / f"random_forest_{direction.lower()}.pkl"
    return DirectionModelState(
        asset=asset,
        direction=direction,
        model_path=path,
        available=_has_file(path),
    )


def inspect_asset_pipeline(asset, raiz_proyecto=None):
    raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]
    asset = asset.upper()
    folder = asset.lower()
    data_dir = raiz / "data" / folder
    results_dir = raiz / "results" / folder
    models_dir = raiz / "models" / folder

    data_path = data_dir / f"{folder}_h1.csv"
    indicators_path = data_dir / f"{folder}_h1_indicators.csv"
    features_path = data_dir / "ml_dataset.csv"
    oof_summary = results_dir / "oof_backtest_summary.csv"

    return AssetPipelineState(
        asset=asset,
        data_path=data_path,
        indicators_path=indicators_path,
        features_path=features_path,
        long_model=_direction_state(raiz, asset, "LONG"),
        short_model=_direction_state(raiz, asset, "SHORT"),
        backtest_ready=_has_file(results_dir / "summary.csv")
        and _has_file(results_dir / "trades.csv"),
        walk_forward_ready=_has_file(results_dir / "walkforward_long.csv")
        and _has_file(results_dir / "walkforward_short.csv"),
        oof_ready=_has_file(results_dir / "oof_predictions.csv")
        and _has_file(oof_summary),
        costs_ready=_has_costs_summary(oof_summary),
        broker_aware_sizing_ready=_has_file(results_dir / "broker_sizing_summary.csv"),
        frozen_model_ready=_has_file(models_dir / "frozen_model_metadata.json"),
        forward_test_ready=_has_file(results_dir / "forward_test.csv"),
    )


def inspect_all_pipelines(raiz_proyecto=None, assets=None):
    return [
        inspect_asset_pipeline(asset, raiz_proyecto)
        for asset in (assets or TARGET_LOGICAL_SYMBOLS)
    ]


def models_available_by_asset(raiz_proyecto=None, assets=None):
    states = inspect_all_pipelines(raiz_proyecto, assets)
    return {
        state.asset: [
            direction.direction
            for direction in (state.long_model, state.short_model)
            if direction.available
        ]
        for state in states
    }
