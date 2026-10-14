import numpy as np
import pytest

from pf_bound.bid_policy import award_probability, choose_bid_price


def test_award_probability_is_share_of_days_at_or_above_bid():
    hist = np.array([10.0, 20.0, 30.0])
    assert award_probability(20.0, hist) == pytest.approx(2 / 3)
    assert award_probability(30.0, hist) == pytest.approx(1 / 3)
    assert award_probability(31.0, hist) == 0.0


def test_bid_maximises_expected_markup():
    hist = np.array([10.0, 20.0, 30.0])
    # (p - 0) * P: 10 * 1, 20 * 2/3, 30 * 1/3 -> 20
    assert choose_bid_price(0.0, hist) == pytest.approx(20.0)
    # (p - 15) * P: 20 -> 5 * 2/3, 30 -> 15 * 1/3 -> 30
    assert choose_bid_price(15.0, hist) == pytest.approx(30.0)


def test_never_bids_below_reservation_price():
    hist = np.array([10.0, 20.0, 30.0])
    assert choose_bid_price(25.0, hist) >= 25.0


def test_no_bid_when_reservation_price_exceeds_history():
    assert choose_bid_price(40.0, np.array([10.0, 20.0, 30.0])) is None


def test_no_bid_without_history():
    assert choose_bid_price(0.0, np.array([])) is None
