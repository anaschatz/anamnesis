from anamnesis.openmemory_vllm_v10 import FIXTURE_PATH, _load_inputs


def test_v10_frozen_matrix_is_fresh_and_stratified() -> None:
    pin, fixture, runtime, sdk = _load_inputs()
    assert len(fixture.cases) == pin.case_count == 12
    assert sum(case.helpful_opportunity for case in fixture.cases) == 6
    assert runtime.served_model == "anamnesis-openmemory-v10"
    assert sdk.package_version == "1.3.0"
    text = FIXTURE_PATH.read_text()
    assert "omsdk8_" not in text
    assert "omsdk9_" not in text
    assert text.count('"id": "omsdk10_') >= 12


def test_v10_cases_have_unique_ids_and_exact_action_evidence() -> None:
    _, fixture, _, _ = _load_inputs()
    ids = [case.id for case in fixture.cases]
    event_ids = [case.event.id for case in fixture.cases]
    assert len(ids) == len(set(ids)) == 12
    assert len(event_ids) == len(set(event_ids)) == 12
    for case in fixture.cases:
        if case.expected.mode == "emit":
            assert case.expected.action_key == case.event.id
            assert case.expected.evidence_event_ids == (case.event.id,)
        else:
            assert case.expected.evidence_event_ids == ()
