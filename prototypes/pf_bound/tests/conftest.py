import pytest

from pf_bound.config import Afrr, Battery, Degradation, Market, Poi, SiteConfig, Solver, Tariff


def make_config(
    *,
    capacity=1.0,
    soc_min=0.0,
    soc_max=1.0,
    p_max=4.0,
    eta=0.9,
    poi_limit=10.0,
    peak_price=0.0,
    volumetric=0.0,
    levies=0.0,
    id_enabled=False,
    id_cost=0.0,
    cycle_limit=None,
    forbid_simultaneous=False,
    corridor_slots=1,
    bid_policy_max_iter=5,
    free_energy_bids=False,
) -> SiteConfig:
    return SiteConfig(
        battery=Battery(capacity, soc_min, soc_max, p_max, p_max, eta, eta),
        poi=Poi(poi_limit, poi_limit),
        tariff=Tariff(peak_price, "month", volumetric, levies),
        market=Market(id_enabled, id_cost),
        degradation=Degradation(cycle_limit),
        afrr=Afrr(corridor_slots, free_energy_bids, 30, bid_policy_max_iter),
        solver=Solver(forbid_simultaneous),
    )


@pytest.fixture
def config_factory():
    return make_config
