"""Ontology engine — dynamic type system with inheritance and constraints."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _prop_list() -> list[PropertyDef]:
    return []


def _str_list() -> list[str]:
    return []


@dataclass(slots=True)
class PropertyDef:
    """A property definition for a node/edge type."""

    name: str
    value_type: str = "str"  # "str", "int", "float", "bool", "list[str]"
    required: bool = False
    default: str = ""


@dataclass(slots=True)
class TypeDef:
    """Defines a node or edge type in the ontology."""

    name: str
    parent: str = ""
    properties: list[PropertyDef] = field(default_factory=_prop_list)
    description: str = ""


@dataclass(slots=True)
class RelationConstraint:
    """Domain/range constraint for an edge kind."""

    edge_kind: str
    domain_types: list[str] = field(default_factory=_str_list)
    range_types: list[str] = field(default_factory=_str_list)


class OntologyRegistry:
    """Runtime ontology registry with type hierarchy, property inheritance, and validation."""

    __slots__ = ("_constraints", "_types")

    def __init__(self) -> None:
        self._types: dict[str, TypeDef] = {}
        self._constraints: list[RelationConstraint] = []

    # --- Type registration ---

    def register_type(self, typedef: TypeDef) -> None:
        """Register or update a type definition."""
        if typedef.parent and typedef.parent not in self._types:
            msg = f"Parent type '{typedef.parent}' not registered"
            raise ValueError(msg)
        self._types[typedef.name] = typedef

    def register_constraint(self, constraint: RelationConstraint) -> None:
        """Register a relation constraint."""
        self._constraints.append(constraint)

    # --- Type queries ---

    def get_type(self, name: str) -> TypeDef | None:
        """Get a type definition by name."""
        return self._types.get(name)

    def all_types(self) -> list[TypeDef]:
        """Return all registered types."""
        return list(self._types.values())

    def get_ancestors(self, name: str) -> list[str]:
        """Walk the is_a chain upward. Returns [parent, grandparent, ...]."""
        ancestors: list[str] = []
        current = name
        visited: set[str] = {current}
        while True:
            td = self._types.get(current)
            if td is None or not td.parent:
                break
            if td.parent in visited:
                break  # cycle guard
            ancestors.append(td.parent)
            visited.add(td.parent)
            current = td.parent
        return ancestors

    def subtypes_of(self, name: str) -> list[str]:
        """Return all direct and indirect subtypes of a type."""
        result: list[str] = []
        for td in self._types.values():
            if td.name == name:
                continue
            if name in self.get_ancestors(td.name) or td.parent == name:
                result.append(td.name)
        return result

    def is_a(self, type_name: str, ancestor: str) -> bool:
        """Check if type_name is a subtype of (or equal to) ancestor."""
        if type_name == ancestor:
            return True
        return ancestor in self.get_ancestors(type_name)

    # --- Property inference (with inheritance) ---

    def infer_properties(self, type_name: str) -> list[PropertyDef]:
        """Get all properties for a type, including inherited ones.

        Properties from ancestors come first; child properties override by name.
        """
        chain = [type_name, *self.get_ancestors(type_name)]
        chain.reverse()  # root → leaf order

        seen: dict[str, PropertyDef] = {}
        for tn in chain:
            td = self._types.get(tn)
            if td is None:
                continue
            for prop in td.properties:
                seen[prop.name] = prop
        return list(seen.values())

    # --- Validation ---

    def validate_node(self, kind: str, properties: dict[str, str]) -> list[str]:
        """Validate node properties against the ontology. Returns list of errors."""
        td = self._types.get(kind)
        if td is None:
            return []  # no type def = no constraints to violate

        errors: list[str] = []
        all_props = self.infer_properties(kind)

        for prop in all_props:
            if prop.required and prop.name not in properties:
                errors.append(f"Missing required property '{prop.name}' for type '{kind}'")

        # Type check
        for prop in all_props:
            if prop.name not in properties:
                continue
            value = properties[prop.name]
            err = _check_type(prop.name, value, prop.value_type)
            if err:
                errors.append(err)

        return errors

    def validate_edge(
        self,
        edge_kind: str,
        source_kind: str,
        target_kind: str,
    ) -> list[str]:
        """Validate edge against relation constraints. Returns list of errors."""
        errors: list[str] = []
        for c in self._constraints:
            if c.edge_kind != edge_kind:
                continue
            if c.domain_types:
                if not any(self.is_a(source_kind, dt) for dt in c.domain_types):
                    errors.append(
                        f"Edge '{edge_kind}': source type '{source_kind}' "
                        f"not in allowed domains {c.domain_types}"
                    )
            if c.range_types:
                if not any(self.is_a(target_kind, rt) for rt in c.range_types):
                    errors.append(
                        f"Edge '{edge_kind}': target type '{target_kind}' "
                        f"not in allowed ranges {c.range_types}"
                    )
        return errors

    def get_constraints_for(self, edge_kind: str) -> list[RelationConstraint]:
        """Get all constraints for a given edge kind."""
        return [c for c in self._constraints if c.edge_kind == edge_kind]

    # --- Serialization helpers ---

    def to_dict(self) -> dict[str, object]:
        """Serialize the registry for export."""
        return {
            "types": [
                {
                    "name": td.name,
                    "parent": td.parent,
                    "description": td.description,
                    "properties": [
                        {
                            "name": p.name,
                            "value_type": p.value_type,
                            "required": p.required,
                            "default": p.default,
                        }
                        for p in td.properties
                    ],
                }
                for td in self._types.values()
            ],
            "constraints": [
                {
                    "edge_kind": c.edge_kind,
                    "domain_types": c.domain_types,
                    "range_types": c.range_types,
                }
                for c in self._constraints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OntologyRegistry:
        """Deserialize a registry from export format."""
        registry = cls()
        types_data = data.get("types", [])
        if not isinstance(types_data, list):
            return registry

        def _parse_typedef(td_raw: object) -> TypeDef | None:
            if not isinstance(td_raw, dict):
                return None
            td_data: dict[str, object] = td_raw
            raw_props = td_data.get("properties", [])
            props: list[PropertyDef] = []
            if isinstance(raw_props, list):
                for p_raw in raw_props:
                    if isinstance(p_raw, dict):
                        props.append(
                            PropertyDef(
                                name=str(p_raw.get("name", "")),
                                value_type=str(p_raw.get("value_type", "str")),
                                required=bool(p_raw.get("required", False)),
                                default=str(p_raw.get("default", "")),
                            )
                        )
            return TypeDef(
                name=str(td_data.get("name", "")),
                parent=str(td_data.get("parent", "")),
                properties=props,
                description=str(td_data.get("description", "")),
            )

        # First pass: register types whose parent is already registered (or has no parent)
        for td_raw in types_data:
            typedef = _parse_typedef(td_raw)
            if typedef is None:
                continue
            if typedef.parent and typedef.parent not in registry._types:
                continue  # defer to second pass
            registry._types[typedef.name] = typedef

        # Second pass: register deferred types
        for td_raw in types_data:
            typedef = _parse_typedef(td_raw)
            if typedef is None:
                continue
            if typedef.name not in registry._types:
                registry._types[typedef.name] = typedef

        constraints_data = data.get("constraints", [])
        if isinstance(constraints_data, list):
            for c_raw in constraints_data:
                if isinstance(c_raw, dict):
                    c_data: dict[str, object] = c_raw
                    domain = c_data.get("domain_types", [])
                    rng = c_data.get("range_types", [])
                    registry._constraints.append(
                        RelationConstraint(
                            edge_kind=str(c_data.get("edge_kind", "")),
                            domain_types=domain if isinstance(domain, list) else [],
                            range_types=rng if isinstance(rng, list) else [],
                        )
                    )
        return registry

    # --- TOML loader ---

    @classmethod
    def load(cls, path: Path | str) -> OntologyRegistry:
        """Load a registry from a TOML file.

        TOML schema uses array-of-tables for types and constraints::

            [[types]]
            name = "knowledge"
            description = "Base type for all knowledge nodes"

            [[types]]
            name = "rule"
            parent = "knowledge"
            description = "Governing constraint or policy"

            [[types.properties]]
            name = "severity"
            value_type = "str"
            required = true
            default = ""

            [[constraints]]
            edge_kind = "learned_from"
            domain_types = ["lesson"]
            range_types = ["outcome", "decision", "agent_activity"]

        This is a thin wrapper around :meth:`from_dict` so the two
        loaders share validation. Use ``load`` when your profile lives
        next to a TOML config (``eval/data/profiles/``) or when you want
        comments in your ontology; use ``from_dict`` when the registry
        was serialized via :meth:`to_dict`.
        """
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)


def _check_type(name: str, value: str, expected: str) -> str:
    """Validate a string value against an expected type. Returns error or empty string."""
    match expected:
        case "int":
            try:
                int(value)
            except ValueError:
                return f"Property '{name}': expected int, got '{value}'"
        case "float":
            try:
                float(value)
            except ValueError:
                return f"Property '{name}': expected float, got '{value}'"
        case "bool":
            if value.lower() not in ("true", "false", "1", "0"):
                return f"Property '{name}': expected bool, got '{value}'"
        case _:
            pass  # "str", "list[str]", etc. — no validation needed
    return ""


def build_agent_ontology() -> OntologyRegistry:
    """Build the default ontology for agent activity tracking.

    Pre-registers types for sessions, tool calls, decisions, outcomes, etc.
    """
    registry = OntologyRegistry()

    # Base types
    registry.register_type(
        TypeDef(
            name="knowledge",
            description="Base type for all knowledge nodes",
        )
    )
    registry.register_type(
        TypeDef(
            name="agent_activity",
            description="Base type for agent-generated activity records",
        )
    )

    # Knowledge subtypes
    for name, desc in [
        ("concept", "Abstract concept or idea"),
        ("entity", "Concrete named entity"),
        ("lesson", "Learned insight from experience"),
        ("decision", "A choice made with rationale"),
        ("rule", "Governing constraint or policy"),
        ("artifact", "Produced output or deliverable"),
    ]:
        registry.register_type(TypeDef(name=name, parent="knowledge", description=desc))

    # Decision has structured properties
    registry.register_type(
        TypeDef(
            name="decision",
            parent="knowledge",
            description="A choice made with rationale",
            properties=[
                PropertyDef(name="rationale", value_type="str", required=True),
                PropertyDef(name="alternatives", value_type="str"),
            ],
        )
    )

    # Activity subtypes
    registry.register_type(
        TypeDef(
            name="session",
            parent="agent_activity",
            description="An agent work session",
            properties=[
                PropertyDef(name="agent_id", value_type="str"),
                PropertyDef(name="start_time", value_type="str"),
                PropertyDef(name="end_time", value_type="str"),
                PropertyDef(name="status", value_type="str", default="active"),
            ],
        )
    )
    registry.register_type(
        TypeDef(
            name="tool_call",
            parent="agent_activity",
            description="An agent tool invocation",
            properties=[
                PropertyDef(name="tool_name", value_type="str", required=True),
                PropertyDef(name="parameters", value_type="str"),
                PropertyDef(name="result_summary", value_type="str"),
                PropertyDef(name="success", value_type="bool"),
                PropertyDef(name="duration_ms", value_type="float"),
            ],
        )
    )
    registry.register_type(
        TypeDef(
            name="observation",
            parent="agent_activity",
            description="Something the agent observed from tool output or environment",
        )
    )
    registry.register_type(
        TypeDef(
            name="reasoning",
            parent="agent_activity",
            description="An agent's reasoning step or chain of thought",
        )
    )
    registry.register_type(
        TypeDef(
            name="outcome",
            parent="agent_activity",
            description="The result of a decision or action",
            properties=[
                PropertyDef(name="success", value_type="bool", required=True),
                PropertyDef(name="impact", value_type="str"),
            ],
        )
    )

    # Relation constraints
    registry.register_constraint(
        RelationConstraint(
            edge_kind="resulted_in",
            domain_types=["decision"],
            range_types=["outcome"],
        )
    )
    registry.register_constraint(
        RelationConstraint(
            edge_kind="invoked",
            domain_types=["session", "agent_activity"],
            range_types=["tool_call"],
        )
    )
    registry.register_constraint(
        RelationConstraint(
            edge_kind="part_of",
            domain_types=["agent_activity"],
            range_types=["session"],
        )
    )
    registry.register_constraint(
        RelationConstraint(
            edge_kind="learned_from",
            domain_types=["lesson"],
            range_types=["outcome", "decision", "agent_activity"],
        )
    )

    return registry
