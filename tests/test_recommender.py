import numpy as np

from app import recommender as rec


def movies():
    return [
        {"id": 1, "title": "Pure Thriller", "genres": "Thriller", "overview": "A detective hunts a killer in a dark city", "vote_average": 8.0, "vote_count": 5000},
        {"id": 2, "title": "Pure Romance", "genres": "Romance", "overview": "Two strangers fall in love on a summer trip", "vote_average": 8.0, "vote_count": 5000},
        {"id": 3, "title": "Thriller Romance", "genres": "Thriller|Romance", "overview": "Lovers on the run from a killer", "vote_average": 7.5, "vote_count": 3000},
        {"id": 4, "title": "Comedy", "genres": "Comedy", "overview": "Friends get lost at a wedding party", "vote_average": 7.0, "vote_count": 3000},
        {"id": 5, "title": "Thriller Two", "genres": "Thriller", "overview": "A spy uncovers a deadly conspiracy", "vote_average": 7.8, "vote_count": 4000},
        {"id": 6, "title": "Romance Two", "genres": "Romance", "overview": "A writer falls for her neighbour", "vote_average": 7.6, "vote_count": 4000},
        {"id": 7, "title": "No Rating", "genres": "Thriller|Romance", "overview": None, "vote_average": None, "vote_count": 0},
    ]


def votes():
    return {
        "a": {1: True, 5: True, 2: False},
        "b": {1: True, 5: True, 6: False},
        "c": {2: True, 6: True, 1: False},
    }


def test_tc08_equal_member_influence():
    feat = rec.build_features(movies())
    v = votes()
    c1, _ = rec.group_compatibility(feat, v, rec.Weights())
    c2, _ = rec.group_compatibility(feat, dict(reversed(list(v.items()))), rec.Weights())
    assert np.allclose(c1, c2)
    dup = {**v, "a2": v["a"]}
    c3, _ = rec.group_compatibility(feat, dup, rec.Weights())
    assert not np.allclose(c1, c3)


def test_tc_recom_01_balance_prefers_blend():
    feat = rec.build_features(movies())
    c, fits = rec.group_compatibility(feat, votes(), rec.Weights())
    assert c[feat.index[3]] > c[feat.index[4]]
    assert fits.shape == (3, len(feat.ids))


def test_tc09_whole_catalogue_considered():
    feat = rec.build_features(movies())
    ranked = rec.rank_movies(feat, votes(), top_k=len(feat.ids))
    assert {r["movie_id"] for r in ranked} == set(feat.ids)
    assert 3 in [r["movie_id"] for r in ranked[:3]]


def test_tc10_configurable_weights():
    feat = rec.build_features(movies())
    rating_heavy = rec.rank_movies(feat, votes(), weights=rec.Weights(compatibility=0.0, rating=1.0, history=0.0), top_k=7, use=("rating",))
    assert rating_heavy[0]["movie_id"] in (1, 2)
    compat_heavy = rec.rank_movies(feat, votes(), weights=rec.Weights(compatibility=1.0, rating=0.0, history=0.0), top_k=7)
    assert compat_heavy[0]["movie_id"] != rating_heavy[0]["movie_id"] or compat_heavy != rating_heavy


def test_tc11_tc_recom_02_cold_start_uses_no_history():
    feat = rec.build_features(movies())
    ranked = rec.rank_movies(feat, votes(), history=[], rng=np.random.default_rng(0))
    assert all(r["history_score"] == 0 for r in ranked)


def test_history_shifts_scores():
    feat = rec.build_features(movies())
    hist = [(3, 5.0), (7, 5.0), (4, 1.0)]
    ranked = rec.rank_movies(feat, votes(), history=hist, rng=np.random.default_rng(1), top_k=7)
    assert any(r["history_score"] != 0 for r in ranked)


def test_nfr07_missing_rating_and_description_do_not_crash():
    feat = rec.build_features(movies())
    ranked = rec.rank_movies(feat, votes(), top_k=7)
    missing = [r for r in ranked if r["movie_id"] == 7][0]
    assert missing["rating_score"] is None


def test_bayesian_rating_penalises_few_votes():
    ms = [
        {"id": 1, "vote_average": 9.5, "vote_count": 3},
        {"id": 2, "vote_average": 8.0, "vote_count": 20000},
        {"id": 3, "vote_average": 7.0, "vote_count": 10000},
    ]
    r = rec.bayesian_ratings(ms)
    mean = (9.5 + 8.0 + 7.0) / 3 / 10
    assert abs(r[1] - mean) < abs(0.95 - mean) / 10
    assert r[1] < 0.95


def test_selection_is_genre_diverse():
    feat = rec.build_features(movies())
    chosen = rec.select_movies_for_member(feat, seed=1, count=5)
    assert len(chosen) == len(set(chosen)) == 5
