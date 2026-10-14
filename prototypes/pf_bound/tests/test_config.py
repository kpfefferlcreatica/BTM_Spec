import textwrap

import pytest

from pf_bound.config import PlaceholderError, load_config

YAML = textwrap.dedent(
    """
    battery:
      energy_capacity_mwh: !placeholder 2.0
      soc_min_mwh: 0.2
      soc_max_mwh: 1.8
      p_charge_max_mw: 1.0
      p_discharge_max_mw: 1.0
      eta_charge: 0.95
      eta_discharge: 0.95
    poi:
      import_limit_mw: 3.0
      export_limit_mw: 3.0
    tariff:
      peak_price_eur_per_mw: 100000.0
      peak_accounting_period: month
      volumetric_charge_eur_per_mwh: 10.0
      levies_eur_per_mwh: 5.0
    market:
      id_enabled: true
      id_transaction_cost_eur_per_mwh: 0.5
    degradation:
      cycle_limit_per_day: 1.5
    afrr:
      corridor_duration_slots: 1
      free_energy_bids: false
      capacity_lookback_days: 30
      bid_policy_max_iter: 5
    solver:
      forbid_simultaneous_charge_discharge: false
    """
)


def write(tmp_path, text):
    path = tmp_path / "site.yaml"
    path.write_text(text)
    return path


def test_placeholder_refused_by_default(tmp_path):
    with pytest.raises(PlaceholderError, match="battery.energy_capacity_mwh"):
        load_config(write(tmp_path, YAML))


def test_placeholder_accepted_when_allowed(tmp_path):
    cfg = load_config(write(tmp_path, YAML), allow_placeholders=True)
    assert cfg.battery.energy_capacity_mwh == 2.0
    assert cfg.placeholders == ("battery.energy_capacity_mwh",)


def test_no_placeholders_loads_clean(tmp_path):
    cfg = load_config(write(tmp_path, YAML.replace("!placeholder ", "")))
    assert cfg.placeholders == ()
    assert cfg.tariff.peak_accounting_period == "month"
    assert cfg.afrr.corridor_duration_slots == 1


def test_soc_band_inside_capacity(tmp_path):
    bad = YAML.replace("!placeholder ", "").replace("soc_max_mwh: 1.8", "soc_max_mwh: 2.5")
    with pytest.raises(ValueError, match="soc_max_mwh"):
        load_config(write(tmp_path, bad))


def test_unknown_key_rejected(tmp_path):
    bad = YAML.replace("!placeholder ", "") + "extra: 1\n"
    with pytest.raises(ValueError, match="extra"):
        load_config(write(tmp_path, bad))
