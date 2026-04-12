from __future__ import annotations

from typing import Dict, List


TASKS: List[Dict] = [
    {
        "task_id": "easy_signal_detection",
        "difficulty": "easy",
        "max_steps": 1,
        "notes": "Read a single-stock technical snapshot and choose the best immediate action.",
        "scenario_steps": [
            {
                "allowed_actions": ["buy", "sell", "hold"],
                "notes": "AAPL is breaking higher with bullish momentum and confirming volume.",
                "market": [
                    {
                        "ticker": "AAPL",
                        "close": 189.30,
                        "ema_20_gap_pct": 3.4,
                        "macd": 1.21,
                        "macd_signal": 0.72,
                        "rsi": 63.0,
                        "stochastic_k": 78.0,
                        "stochastic_d": 71.0,
                        "bb_pos": 0.74,
                        "volume_zscore": 1.2,
                        "obv_slope": 0.9,
                        "trend_label": "up",
                        "volatility_label": "medium",
                    }
                ],
                "portfolio": {
                    "cash": 100000.0,
                    "total_value": 100000.0,
                    "max_drawdown_pct": 15.0,
                    "current_drawdown_pct": 0.0,
                    "exposure_pct": 0.0,
                    "cooldown_remaining": 0,
                },
                "positions": {},
                "gold": {
                    "action_type": "buy",
                    "ticker": "AAPL",
                    "confidence_band": [0.60, 0.95],
                    "rationale_tags": ["trend_up", "macd_bullish", "volume_confirmation"],
                },
            }
        ],
    },
    {
        "task_id": "medium_position_management",
        "difficulty": "medium",
        "max_steps": 2,
        "notes": "Manage an existing position under drawdown pressure, then react to the next market state.",
        "scenario_steps": [
            {
                "allowed_actions": ["sell", "hold", "reduce"],
                "notes": "MSFT is in a sharp downtrend, drawdown is widening, and the desk needs immediate risk control.",
                "market": [
                    {
                        "ticker": "MSFT",
                        "close": 327.50,
                        "ema_20_gap_pct": -2.1,
                        "macd": -0.88,
                        "macd_signal": -0.42,
                        "rsi": 34.0,
                        "stochastic_k": 19.0,
                        "stochastic_d": 24.0,
                        "bb_pos": 0.18,
                        "volume_zscore": 1.7,
                        "obv_slope": -0.7,
                        "trend_label": "down",
                        "volatility_label": "high",
                    }
                ],
                "portfolio": {
                    "cash": 15000.0,
                    "total_value": 100000.0,
                    "max_drawdown_pct": 15.0,
                    "current_drawdown_pct": 8.0,
                    "exposure_pct": 70.0,
                    "cooldown_remaining": 0,
                },
                "positions": {
                    "MSFT": {
                        "shares_held": 200,
                        "entry_price": 352.0,
                        "market_value": 65500.0,
                        "unrealized_pnl_pct": -6.96,
                    }
                },
                "gold": {
                    "action_type": "sell",
                    "ticker": "MSFT",
                    "order_fraction_band": [0.90, 1.0],
                    "confidence_band": [0.65, 1.0],
                    "rationale_tags": ["stop_loss", "downtrend", "high_volatility"],
                },
            },
            {
                "allowed_actions": ["hold", "buy", "sell"],
                "notes": "After de-risking, MSFT stabilizes on lower volatility. The desk should avoid impulsive re-entry.",
                "market": [
                    {
                        "ticker": "MSFT",
                        "close": 331.20,
                        "ema_20_gap_pct": -0.4,
                        "macd": -0.12,
                        "macd_signal": -0.22,
                        "rsi": 47.0,
                        "stochastic_k": 44.0,
                        "stochastic_d": 41.0,
                        "bb_pos": 0.42,
                        "volume_zscore": -0.2,
                        "obv_slope": 0.1,
                        "trend_label": "sideways",
                        "volatility_label": "medium",
                    }
                ],
                "portfolio": {
                    "cash": 80500.0,
                    "total_value": 100000.0,
                    "max_drawdown_pct": 15.0,
                    "current_drawdown_pct": 4.0,
                    "exposure_pct": 19.5,
                    "cooldown_remaining": 1,
                },
                "positions": {
                    "MSFT": {
                        "shares_held": 60,
                        "entry_price": 352.0,
                        "market_value": 19872.0,
                        "unrealized_pnl_pct": -5.91,
                    }
                },
                "gold": {
                    "action_type": "hold",
                    "ticker": "MSFT",
                    "order_fraction_band": [0.0, 0.2],
                    "confidence_band": [0.45, 0.85],
                    "rationale_tags": ["capital_preservation", "wait_for_confirmation", "reduced_risk"],
                },
            },
        ],
    },
    {
        "task_id": "hard_portfolio_allocation",
        "difficulty": "hard",
        "max_steps": 2,
        "notes": "Allocate across a watchlist, then adapt the book after a volatility spike.",
        "constraints": {
            "max_single_name_weight": 0.50,
            "min_cash_reserve": 0.10,
        },
        "scenario_steps": [
            {
                "allowed_actions": ["rebalance", "hold"],
                "notes": "Build a fresh portfolio using three names while respecting concentration and reserve rules.",
                "market": [
                    {
                        "ticker": "NVDA",
                        "close": 902.0,
                        "ema_20_gap_pct": 4.8,
                        "macd": 2.4,
                        "macd_signal": 1.7,
                        "rsi": 67.0,
                        "stochastic_k": 80.0,
                        "stochastic_d": 74.0,
                        "bb_pos": 0.79,
                        "volume_zscore": 1.3,
                        "obv_slope": 1.0,
                        "trend_label": "up",
                        "volatility_label": "high",
                    },
                    {
                        "ticker": "META",
                        "close": 486.0,
                        "ema_20_gap_pct": 2.5,
                        "macd": 1.0,
                        "macd_signal": 0.8,
                        "rsi": 58.0,
                        "stochastic_k": 60.0,
                        "stochastic_d": 57.0,
                        "bb_pos": 0.63,
                        "volume_zscore": 0.8,
                        "obv_slope": 0.7,
                        "trend_label": "up",
                        "volatility_label": "medium",
                    },
                    {
                        "ticker": "JNJ",
                        "close": 151.0,
                        "ema_20_gap_pct": 0.6,
                        "macd": 0.1,
                        "macd_signal": 0.12,
                        "rsi": 50.0,
                        "stochastic_k": 49.0,
                        "stochastic_d": 47.0,
                        "bb_pos": 0.51,
                        "volume_zscore": -0.1,
                        "obv_slope": 0.1,
                        "trend_label": "sideways",
                        "volatility_label": "low",
                    },
                ],
                "portfolio": {
                    "cash": 100000.0,
                    "total_value": 100000.0,
                    "max_drawdown_pct": 15.0,
                    "current_drawdown_pct": 2.0,
                    "exposure_pct": 0.0,
                    "cooldown_remaining": 0,
                },
                "positions": {},
                "gold": {
                    "top_pick": "NVDA",
                    "allocation_bands": {
                        "NVDA": [0.35, 0.50],
                        "META": [0.20, 0.35],
                        "JNJ": [0.05, 0.20],
                    },
                    "cash_reserve_band": [0.10, 0.25],
                    "rationale_tags": ["momentum_leader", "risk_balanced", "cash_buffer"],
                },
            },
            {
                "allowed_actions": ["rebalance", "hold"],
                "notes": "NVDA becomes more volatile after a headline spike. The desk should trim concentration and keep cash ready.",
                "market": [
                    {
                        "ticker": "NVDA",
                        "close": 944.0,
                        "ema_20_gap_pct": 6.3,
                        "macd": 2.7,
                        "macd_signal": 2.0,
                        "rsi": 74.0,
                        "stochastic_k": 89.0,
                        "stochastic_d": 82.0,
                        "bb_pos": 0.91,
                        "volume_zscore": 2.1,
                        "obv_slope": 1.4,
                        "trend_label": "up",
                        "volatility_label": "high",
                    },
                    {
                        "ticker": "META",
                        "close": 492.0,
                        "ema_20_gap_pct": 2.8,
                        "macd": 1.1,
                        "macd_signal": 0.9,
                        "rsi": 60.0,
                        "stochastic_k": 63.0,
                        "stochastic_d": 59.0,
                        "bb_pos": 0.66,
                        "volume_zscore": 0.7,
                        "obv_slope": 0.8,
                        "trend_label": "up",
                        "volatility_label": "medium",
                    },
                    {
                        "ticker": "JNJ",
                        "close": 152.0,
                        "ema_20_gap_pct": 0.9,
                        "macd": 0.14,
                        "macd_signal": 0.12,
                        "rsi": 52.0,
                        "stochastic_k": 51.0,
                        "stochastic_d": 48.0,
                        "bb_pos": 0.53,
                        "volume_zscore": 0.1,
                        "obv_slope": 0.2,
                        "trend_label": "sideways",
                        "volatility_label": "low",
                    },
                ],
                "portfolio": {
                    "cash": 15000.0,
                    "total_value": 104000.0,
                    "max_drawdown_pct": 15.0,
                    "current_drawdown_pct": 3.5,
                    "exposure_pct": 85.0,
                    "cooldown_remaining": 0,
                },
                "positions": {
                    "NVDA": {
                        "shares_held": 48,
                        "entry_price": 902.0,
                        "market_value": 45312.0,
                        "unrealized_pnl_pct": 4.66,
                    },
                    "META": {
                        "shares_held": 60,
                        "entry_price": 486.0,
                        "market_value": 29520.0,
                        "unrealized_pnl_pct": 1.23,
                    },
                    "JNJ": {
                        "shares_held": 75,
                        "entry_price": 151.0,
                        "market_value": 11400.0,
                        "unrealized_pnl_pct": 0.66,
                    },
                },
                "gold": {
                    "top_pick": "NVDA",
                    "allocation_bands": {
                        "NVDA": [0.30, 0.42],
                        "META": [0.22, 0.34],
                        "JNJ": [0.10, 0.20],
                    },
                    "cash_reserve_band": [0.14, 0.28],
                    "rationale_tags": ["trim_winner", "risk_balanced", "cash_buffer"],
                },
            },
        ],
    },
]


def list_tasks() -> List[Dict]:
    return TASKS


def get_task(task_id: str) -> Dict:
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def get_default_task_id() -> str:
    from app import config
    return config.DEFAULT_TASK
