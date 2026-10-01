from src.bot.sizing import compute_contracts

MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50  # $0.50/tick on Micro E-mini Nasdaq-100


def test_normal_case_clamped_to_cap():
    # 10.00 pt stop (40 ticks) -> $20/contract risk. $125/$20 = 6 contracts,
    # but cap is 5 -> clamped to 5, actual risk taken = $100 (under target).
    contracts = compute_contracts(
        entry=21050.00, stop=21040.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=5,
    )
    assert contracts == 5


def test_tail_case_tight_stop_clamped_not_skipped():
    # 1-tick stop -> $0.50/contract risk. $125/$0.50 = 250 contracts,
    # clamped to the cap of 5 rather than skipped or placed at 250.
    contracts = compute_contracts(
        entry=21050.00, stop=21049.75,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=5,
    )
    assert contracts == 5


def test_under_cap_uses_raw_size():
    # 40-tick stop -> $20/contract. $125/$20 = 6, cap is 10 -> not clamped.
    contracts = compute_contracts(
        entry=21050.00, stop=21040.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=10,
    )
    assert contracts == 6


def test_wide_stop_exceeding_risk_target_skips_trade():
    # A single contract's risk already exceeds risk_usd -> skip (0), rather
    # than silently accepting more risk than asked for.
    contracts = compute_contracts(
        entry=21050.00, stop=20950.00,  # 100 pt stop = 400 ticks = $200/contract
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=5,
    )
    assert contracts == 0


def test_zero_stop_distance_returns_zero():
    contracts = compute_contracts(
        entry=21050.00, stop=21050.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=5,
    )
    assert contracts == 0


def test_invalid_tick_or_cap_params_return_zero():
    assert compute_contracts(21050, 21040, 0, MNQ_TICK_VALUE, 125, 5) == 0
    assert compute_contracts(21050, 21040, MNQ_TICK_SIZE, 0, 125, 5) == 0
    assert compute_contracts(21050, 21040, MNQ_TICK_SIZE, MNQ_TICK_VALUE, 125, 0) == 0


def test_moderate_wide_stop_floors_to_one_within_ceiling():
    # 10-pt stop (40 ticks) -> $20/contract. risk_usd=15 can't afford even 1
    # (floor(15/20)=0), but $20 <= max_risk_per_trade_usd=30 -> floors to 1
    # rather than skipping the trade.
    contracts = compute_contracts(
        entry=21050.00, stop=21040.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=15, max_contracts=5, max_risk_per_trade_usd=30,
    )
    assert contracts == 1


def test_too_wide_even_for_ceiling_still_skips():
    # 16-pt stop (64 ticks) -> $32/contract, exceeds max_risk_per_trade_usd=30
    # too -> skipped (0), not forced through at an even bigger overshoot.
    contracts = compute_contracts(
        entry=21050.00, stop=21034.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=15, max_contracts=5, max_risk_per_trade_usd=30,
    )
    assert contracts == 0


def test_no_ceiling_given_keeps_old_skip_behavior():
    # max_risk_per_trade_usd defaults to None -> any stop risk_usd can't
    # afford is skipped outright, same as before this feature existed.
    contracts = compute_contracts(
        entry=21050.00, stop=21040.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=15, max_contracts=5,
    )
    assert contracts == 0


def test_short_direction_uses_absolute_distance():
    # Stop above entry (short trade) - same math, direction-agnostic.
    contracts = compute_contracts(
        entry=21040.00, stop=21050.00,
        tick_size=MNQ_TICK_SIZE, tick_value=MNQ_TICK_VALUE,
        risk_usd=125, max_contracts=5,
    )
    assert contracts == 5
