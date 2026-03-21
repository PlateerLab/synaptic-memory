"""Tests for the ontology engine."""

import pytest

from synaptic.ontology import (
    OntologyRegistry,
    PropertyDef,
    RelationConstraint,
    TypeDef,
    build_agent_ontology,
)


@pytest.fixture
def registry() -> OntologyRegistry:
    """Build a small test ontology."""
    r = OntologyRegistry()
    r.register_type(TypeDef(name="knowledge", description="Base"))
    r.register_type(TypeDef(
        name="concept", parent="knowledge", description="Abstract concept",
    ))
    r.register_type(TypeDef(
        name="lesson", parent="knowledge", description="Learned insight",
        properties=[PropertyDef(name="source_event", value_type="str", required=True)],
    ))
    r.register_type(TypeDef(
        name="decision", parent="knowledge", description="A choice",
        properties=[
            PropertyDef(name="rationale", value_type="str", required=True),
            PropertyDef(name="confidence", value_type="float"),
        ],
    ))
    r.register_type(TypeDef(
        name="technical_decision", parent="decision",
        description="A technical architecture choice",
        properties=[PropertyDef(name="tech_stack", value_type="str")],
    ))
    return r


class TestTypeRegistration:
    def test_register_and_get(self, registry: OntologyRegistry) -> None:
        td = registry.get_type("concept")
        assert td is not None
        assert td.parent == "knowledge"

    def test_register_unknown_parent_raises(self) -> None:
        r = OntologyRegistry()
        with pytest.raises(ValueError, match="Parent type 'nonexistent' not registered"):
            r.register_type(TypeDef(name="child", parent="nonexistent"))

    def test_all_types(self, registry: OntologyRegistry) -> None:
        names = {t.name for t in registry.all_types()}
        assert names == {"knowledge", "concept", "lesson", "decision", "technical_decision"}


class TestTypeHierarchy:
    def test_ancestors(self, registry: OntologyRegistry) -> None:
        assert registry.get_ancestors("technical_decision") == ["decision", "knowledge"]

    def test_ancestors_root(self, registry: OntologyRegistry) -> None:
        assert registry.get_ancestors("knowledge") == []

    def test_subtypes_of(self, registry: OntologyRegistry) -> None:
        subs = registry.subtypes_of("knowledge")
        assert set(subs) == {"concept", "lesson", "decision", "technical_decision"}

    def test_subtypes_of_decision(self, registry: OntologyRegistry) -> None:
        subs = registry.subtypes_of("decision")
        assert subs == ["technical_decision"]

    def test_is_a(self, registry: OntologyRegistry) -> None:
        assert registry.is_a("technical_decision", "knowledge") is True
        assert registry.is_a("technical_decision", "decision") is True
        assert registry.is_a("technical_decision", "technical_decision") is True
        assert registry.is_a("concept", "decision") is False


class TestPropertyInheritance:
    def test_infer_properties_leaf(self, registry: OntologyRegistry) -> None:
        props = registry.infer_properties("technical_decision")
        names = [p.name for p in props]
        # decision's props + technical_decision's props
        assert "rationale" in names
        assert "confidence" in names
        assert "tech_stack" in names

    def test_infer_properties_no_parent(self, registry: OntologyRegistry) -> None:
        props = registry.infer_properties("knowledge")
        assert props == []

    def test_child_overrides_parent_property(self) -> None:
        r = OntologyRegistry()
        r.register_type(TypeDef(
            name="base",
            properties=[PropertyDef(name="x", value_type="str", required=False)],
        ))
        r.register_type(TypeDef(
            name="child", parent="base",
            properties=[PropertyDef(name="x", value_type="int", required=True)],
        ))
        props = r.infer_properties("child")
        assert len(props) == 1
        assert props[0].value_type == "int"
        assert props[0].required is True


class TestValidation:
    def test_validate_node_ok(self, registry: OntologyRegistry) -> None:
        errors = registry.validate_node("decision", {"rationale": "good reason"})
        assert errors == []

    def test_validate_node_missing_required(self, registry: OntologyRegistry) -> None:
        errors = registry.validate_node("decision", {})
        assert any("rationale" in e for e in errors)

    def test_validate_node_bad_type(self, registry: OntologyRegistry) -> None:
        errors = registry.validate_node("decision", {
            "rationale": "ok",
            "confidence": "not-a-float",
        })
        assert any("confidence" in e for e in errors)

    def test_validate_node_unknown_type(self, registry: OntologyRegistry) -> None:
        errors = registry.validate_node("unknown_type", {"foo": "bar"})
        assert errors == []  # no constraints = no errors

    def test_validate_inherited_properties(self, registry: OntologyRegistry) -> None:
        # technical_decision inherits rationale (required) from decision
        errors = registry.validate_node("technical_decision", {"tech_stack": "Python"})
        assert any("rationale" in e for e in errors)


class TestRelationConstraints:
    def test_validate_edge_ok(self, registry: OntologyRegistry) -> None:
        registry.register_constraint(RelationConstraint(
            edge_kind="resulted_in",
            domain_types=["decision"],
            range_types=["concept", "lesson"],
        ))
        errors = registry.validate_edge("resulted_in", "decision", "lesson")
        assert errors == []

    def test_validate_edge_bad_domain(self, registry: OntologyRegistry) -> None:
        registry.register_constraint(RelationConstraint(
            edge_kind="resulted_in",
            domain_types=["decision"],
            range_types=["concept"],
        ))
        errors = registry.validate_edge("resulted_in", "concept", "concept")
        assert any("domain" in e for e in errors)

    def test_validate_edge_subtype_ok(self, registry: OntologyRegistry) -> None:
        registry.register_constraint(RelationConstraint(
            edge_kind="resulted_in",
            domain_types=["decision"],
            range_types=["knowledge"],
        ))
        # technical_decision is_a decision, so it should pass
        errors = registry.validate_edge("resulted_in", "technical_decision", "lesson")
        assert errors == []

    def test_no_constraint_passes(self, registry: OntologyRegistry) -> None:
        errors = registry.validate_edge("random_edge", "concept", "lesson")
        assert errors == []


class TestSerialization:
    def test_round_trip(self, registry: OntologyRegistry) -> None:
        data = registry.to_dict()
        restored = OntologyRegistry.from_dict(data)
        assert len(restored.all_types()) == len(registry.all_types())
        for td in registry.all_types():
            rt = restored.get_type(td.name)
            assert rt is not None
            assert rt.parent == td.parent

    def test_from_dict_empty(self) -> None:
        r = OntologyRegistry.from_dict({})
        assert r.all_types() == []


class TestAgentOntology:
    def test_build_agent_ontology(self) -> None:
        reg = build_agent_ontology()
        assert reg.get_type("session") is not None
        assert reg.get_type("tool_call") is not None
        assert reg.get_type("outcome") is not None

    def test_agent_ontology_hierarchy(self) -> None:
        reg = build_agent_ontology()
        assert reg.is_a("session", "agent_activity")
        assert reg.is_a("tool_call", "agent_activity")
        assert reg.is_a("outcome", "agent_activity")

    def test_agent_ontology_constraints(self) -> None:
        reg = build_agent_ontology()
        # resulted_in: decision → outcome
        errors = reg.validate_edge("resulted_in", "decision", "outcome")
        assert errors == []
        # resulted_in: concept → outcome should fail
        errors = reg.validate_edge("resulted_in", "concept", "outcome")
        assert len(errors) > 0
