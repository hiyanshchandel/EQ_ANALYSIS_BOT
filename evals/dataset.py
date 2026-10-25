"""The golden dataset: realistic user messages with the answer we expect.

Each case is a whole conversation (`turns`); only the LAST turn is graded. Expected
values are only the things that have one right answer (intent, symbol, horizon); the
quality of free text is graded by the LLM judge instead.
"""

from __future__ import annotations

from langfuse.experiment import LocalExperimentItem

EVAL_CASES: list[LocalExperimentItem] = [
    {
        "input": {"turns": ["Analyze Tata Consultancy Services"]},
        "expected_output": {"intent": "analyze", "symbol": "TCS.NS"},
    },
    {
        "input": {"turns": ["how is infosys looking for the long term?"]},
        "expected_output": {"intent": "analyze", "symbol": "INFY.NS", "horizon": "long"},
    },
    {
        "input": {"turns": ["Should I buy HDFC Bank?"]},
        "expected_output": {"intent": "analyze", "symbol": "HDFCBANK.NS"},
    },
    {
        "input": {"turns": ["reliance industries, short term view pls"]},
        "expected_output": {"intent": "analyze", "symbol": "RELIANCE.NS", "horizon": "short"},
    },
    {
        "input": {"turns": ["ITC.NS"]},
        "expected_output": {"intent": "analyze", "symbol": "ITC.NS"},
    },
    {
        "input": {"turns": ["hey, what can you do?"]},
        "expected_output": {"intent": "other"},
    },
    {
        "input": {"turns": ["Analyze Infosys", "why that recommendation?"]},
        "expected_output": {"intent": "follow_up"},
    },
    {
        "input": {"turns": ["Analyze Infosys", "ok now do wipro"]},
        "expected_output": {"intent": "analyze", "symbol": "WIPRO.NS"},
    },
]
