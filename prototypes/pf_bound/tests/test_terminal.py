import pytest

from pf_bound.terminal import terminal_value


def test_mean_of_slots_needed_to_empty_the_band(config_factory):
    cfg = config_factory(soc_min=0.0, soc_max=1.0, p_max=2.0, eta=1.0)
    # 1 MWh at 2 MW over 0.25 h slots -> 2 slots -> mean(40, 30)
    assert terminal_value([10, 20, 30, 40], cfg, slot_hours=0.25) == pytest.approx(35.0)


def test_discharge_efficiency_scales_value_and_slot_count(config_factory):
    cfg = config_factory(soc_min=0.0, soc_max=1.0, p_max=2.0, eta=0.9)
    # AC energy 0.9 MWh -> ceil(0.9 / 0.5) = 2 slots -> 0.9 * 35
    assert terminal_value([10, 20, 30, 40], cfg, slot_hours=0.25) == pytest.approx(31.5)


def test_slot_count_capped_at_horizon(config_factory):
    cfg = config_factory(soc_min=0.0, soc_max=1.0, p_max=0.1, eta=1.0)
    assert terminal_value([10, 30], cfg, slot_hours=0.25) == pytest.approx(20.0)
