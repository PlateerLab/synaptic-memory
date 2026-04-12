"""Generate an interactive HTML graph explorer from a synaptic-memory graph.

Produces a single self-contained HTML file with:
- 2D force-directed graph (force-graph library, Canvas-based, 50K+ nodes)
- Click-to-inspect sidebar (title, content, properties, edges)
- NodeKind color legend + filter checkboxes
- Category dropdown filter
- Title search box with zoom-to-node
- Node size by connection count, edge color by EdgeKind

Usage::

    uv run python eval/scripts/visualize_graph.py
    uv run python eval/scripts/visualize_graph.py --graph eval/data/krra_graph.sqlite
    uv run python eval/scripts/visualize_graph.py --max-nodes 5000 --output my_graph.html

    # Category subset only (fast for large graphs)
    uv run python eval/scripts/visualize_graph.py --category "규정 및 지침"

Opens in browser automatically after generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.models import EdgeKind, NodeKind

DEFAULT_SQLITE = REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"

# Color palette by NodeKind (used when no category color exists)
KIND_COLORS = {
    str(NodeKind.CONCEPT): "#e74c3c",  # red — categories
    str(NodeKind.RULE): "#e67e22",  # orange
    str(NodeKind.DECISION): "#3498db",  # blue
    str(NodeKind.OBSERVATION): "#2ecc71",  # green
    str(NodeKind.OUTCOME): "#9b59b6",  # purple
    str(NodeKind.ARTIFACT): "#1abc9c",  # teal
    str(NodeKind.ENTITY): "#34495e",  # dark gray
    str(NodeKind.CHUNK): "#bdc3c7",  # light gray
    str(NodeKind.COMMUNITY): "#f39c12",  # yellow
}

EDGE_COLORS = {
    str(EdgeKind.PART_OF): "#e74c3c",
    str(EdgeKind.CONTAINS): "#3498db",
    str(EdgeKind.NEXT_CHUNK): "#95a5a6",
    str(EdgeKind.MENTIONS): "#2ecc71",
    str(EdgeKind.RELATED): "#9b59b6",
    str(EdgeKind.SUPERSEDES): "#e67e22",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--graph", type=Path, default=DEFAULT_SQLITE)
    p.add_argument("--backend", choices=["sqlite", "kuzu"], default="sqlite")
    p.add_argument("--category", default=None, help="Filter to a single category")
    p.add_argument("--max-nodes", type=int, default=2000, help="Max nodes to render")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "eval" / "results" / "graph.html")
    p.add_argument("--no-open", action="store_true", help="Don't open browser")
    return p.parse_args()


async def _load_graph_data(args) -> dict:
    """Load nodes and edges from backend into a JSON-friendly dict."""
    if args.backend == "sqlite":
        from synaptic.backends.sqlite_graph import SqliteGraphBackend

        backend = SqliteGraphBackend(str(args.graph))
    else:
        from synaptic.backends.kuzu import KuzuBackend

        backend = KuzuBackend(str(args.graph))

    await backend.connect()

    all_nodes = await backend.list_nodes(kind=None, limit=100_000)

    # Filter by category if specified — include category node + docs + their chunks
    if args.category:
        cat_lower = args.category.lower()
        doc_ids: set[str] = set()
        filtered = []
        all_node_map = {n.id: n for n in all_nodes}

        # Pass 1: find category + document nodes
        for n in all_nodes:
            props = n.properties or {}
            node_cat = (props.get("category") or "").lower()
            if cat_lower in node_cat:
                filtered.append(n)
                if "document" in (n.tags or []):
                    doc_ids.add(n.id)
            elif "category" in (n.tags or []) and cat_lower in n.title.lower():
                filtered.append(n)

        # Pass 2: include chunks belonging to matched documents (via CONTAINS edges)
        for doc_id in doc_ids:
            try:
                edges = await backend.get_edges(doc_id, direction="outgoing")
                for e in edges:
                    if str(e.kind) == str(EdgeKind.CONTAINS) and e.target_id in all_node_map:
                        chunk = all_node_map[e.target_id]
                        if chunk not in filtered:
                            filtered.append(chunk)
            except Exception:
                pass

        all_nodes = filtered

    # Limit node count
    if len(all_nodes) > args.max_nodes:
        # Prioritize: categories first, then documents, then chunks
        priority = {str(NodeKind.CONCEPT): 0, str(NodeKind.ENTITY): 1, str(NodeKind.CHUNK): 2}
        all_nodes.sort(key=lambda n: priority.get(str(n.kind), 1))
        all_nodes = all_nodes[: args.max_nodes]

    node_ids = {n.id for n in all_nodes}

    # Build node list
    nodes = []
    categories = set()
    kinds = set()
    for n in all_nodes:
        props = n.properties or {}
        cat = props.get("category", "")
        if cat:
            categories.add(cat)
        kinds.add(str(n.kind))
        nodes.append(
            {
                "id": n.id,
                "title": n.title[:80],
                "kind": str(n.kind),
                "category": cat,
                "content": (n.content or "")[:500],
                "tags": list(n.tags or []),
                "properties": {k: str(v)[:100] for k, v in (props or {}).items()},
                "color": KIND_COLORS.get(str(n.kind), "#999"),
                "val": 30
                if str(n.kind) == str(NodeKind.CONCEPT)
                else (10 if "document" in (n.tags or []) else 3),
            }
        )

    # Build doc → chunks map for sidebar display
    doc_chunks: dict[str, list[dict]] = {}
    for n in all_nodes:
        if str(n.kind) == str(NodeKind.CHUNK):
            doc_id = (n.properties or {}).get("doc_id", "")
            if doc_id:
                doc_chunks.setdefault(doc_id, []).append(
                    {
                        "index": int((n.properties or {}).get("chunk_index", "0") or "0"),
                        "title": n.title[:60],
                        "content": (n.content or "")[:300],
                    }
                )
    for chunks in doc_chunks.values():
        chunks.sort(key=lambda c: c["index"])

    # Load edges (only between non-chunk visible nodes for graph display)
    links = []
    for n in all_nodes:
        if str(n.kind) == str(NodeKind.CHUNK):
            continue  # skip chunk edges in the graph — too noisy
        try:
            edges = await backend.get_edges(n.id, direction="outgoing")
        except Exception:
            continue
        for e in edges:
            if e.target_id in node_ids:
                target_node = next((nd for nd in all_nodes if nd.id == e.target_id), None)
                if target_node and str(target_node.kind) == str(NodeKind.CHUNK):
                    continue  # skip edges to chunks
                links.append(
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "kind": str(e.kind),
                        "color": EDGE_COLORS.get(str(e.kind), "#ccc"),
                    }
                )

    # Filter out chunk nodes from graph display (they appear in sidebar only)
    graph_nodes = [n for n in nodes if n["kind"] != str(NodeKind.CHUNK)]
    graph_kinds = sorted({n["kind"] for n in graph_nodes})

    await backend.close()

    return {
        "nodes": graph_nodes,
        "links": links,
        "categories": sorted(categories),
        "kinds": graph_kinds,
        "doc_chunks": doc_chunks,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Synaptic Memory — Graph Explorer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; overflow: hidden; background: #1a1a2e; color: #eee; }
#cy { flex: 1; }
#sidebar { width: 380px; background: #16213e; border-left: 1px solid #0f3460; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
#controls { position: absolute; top: 12px; left: 12px; z-index: 10; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
#controls input, #controls select { padding: 6px 10px; border-radius: 6px; border: 1px solid #0f3460; background: #16213e; color: #eee; font-size: 13px; }
#controls input { width: 220px; }
#controls button { padding: 5px 12px; border-radius: 6px; border: 1px solid #0f3460; background: #16213e; color: #eee; cursor: pointer; font-size: 12px; }
#controls button:hover { background: #0f3460; }
.filter-btn { padding: 4px 10px; border-radius: 12px; border: 1px solid #444; background: transparent; color: #eee; cursor: pointer; font-size: 12px; transition: all 0.2s; }
.filter-btn.active { background: var(--c); color: #fff; border-color: var(--c); }
#sidebar h3 { color: #e94560; font-size: 14px; border-bottom: 1px solid #0f3460; padding-bottom: 6px; }
#sidebar .label { color: #888; font-size: 11px; text-transform: uppercase; margin-top: 8px; }
#sidebar .value { font-size: 13px; line-height: 1.5; word-break: break-all; }
#sidebar .content-box { background: #0f3460; padding: 10px; border-radius: 6px; font-size: 12px; line-height: 1.6; max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
#sidebar .edge-list { font-size: 12px; }
#sidebar .edge-item { padding: 4px 0; border-bottom: 1px solid #1a1a2e; cursor: pointer; }
#sidebar .edge-item:hover { color: #e94560; }
#legend { position: absolute; bottom: 12px; left: 12px; background: rgba(22,33,62,0.95); padding: 12px; border-radius: 8px; font-size: 11px; z-index: 10; }
.legend-item { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; }
#stats { position: absolute; bottom: 12px; right: 392px; background: rgba(22,33,62,0.95); padding: 8px 12px; border-radius: 8px; font-size: 11px; color: #888; z-index: 10; }
#placeholder { color: #555; text-align: center; margin-top: 40px; font-size: 13px; }
</style>
</head>
<body>
<div id="cy">
  <div id="controls">
    <input id="search" type="text" placeholder="🔍 노드 검색...">
    <select id="cat-filter"><option value="">전체 카테고리</option></select>
    <button onclick="relayout()">↻ 재정렬</button>
    <button onclick="cy.fit(undefined,50)">⊡ 맞춤</button>
  </div>
  <div id="legend"></div>
  <div id="stats"></div>
</div>
<div id="sidebar">
  <h3>Synaptic Memory Explorer</h3>
  <div id="placeholder">← 노드를 클릭하면 상세 정보가 표시됩니다</div>
  <div id="detail" style="display:none"></div>
</div>
<script src="https://unpkg.com/cytoscape@3/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2/cytoscape-dagre.js"></script>
<script>
const DATA = __GRAPH_DATA__;
const KIND_COLORS = __KIND_COLORS__;
const EDGE_COLORS = __EDGE_COLORS__;
const DOC_CHUNKS = __DOC_CHUNKS__;

// Build all elements at once
const nodeMap = {};
DATA.nodes.forEach(n => { nodeMap[n.id] = n; });

const catColors = {};
DATA.categories.forEach((c, i) => {
  catColors[c] = ['#e74c3c','#3498db','#2ecc71','#e67e22','#9b59b6','#1abc9c','#f39c12','#e91e63','#00bcd4','#8bc34a'][i % 10];
});

const elements = [];
DATA.nodes.forEach(n => {
  const isCat = n.kind === 'concept';
  const docCount = isCat ? DATA.links.filter(l => l.target === n.id).length : 0;
  elements.push({
    data: {
      id: n.id,
      label: isCat ? (n.title + ' (' + docCount + ')') : (n.title?.substring(0, 28) || n.id),
      fullTitle: n.title, kind: n.kind, category: n.category,
      content: n.content, tags: n.tags, properties: n.properties,
      color: isCat ? (catColors[n.title] || '#e74c3c') : (catColors[n.category] || n.color),
    },
    classes: n.kind,
  });
});
DATA.links.forEach((l, i) => {
  elements.push({
    data: { id: 'e' + i, source: l.source, target: l.target, kind: l.kind, color: catColors[nodeMap[l.source]?.category] || l.color },
    classes: l.kind,
  });
});

// Cytoscape instance
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    // --- Nodes ---
    { selector: 'node', style: {
      'label': '',
      'background-color': 'data(color)',
      'border-width': 0,
      'width': 10, 'height': 10,
    }},
    // Category — always labeled, big
    { selector: 'node.concept', style: {
      'width': 100, 'height': 100,
      'label': 'data(label)',
      'font-size': '13px',
      'font-weight': 'bold',
      'color': '#fff',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '80px',
      'border-width': 3,
      'border-color': '#e74c3c',
      'text-background-color': '#1a1a2e',
      'text-background-opacity': 0.9,
      'text-background-padding': '4px',
      'text-background-shape': 'roundrectangle',
    }},
    // Documents — medium circle, label on hover
    { selector: 'node.rule, node.decision, node.observation, node.outcome, node.artifact, node.entity', style: {
      'width': 30, 'height': 30,
      'border-width': 2,
      'border-color': 'data(color)',
      'background-opacity': 0.8,
    }},
    // Hover — show label
    { selector: 'node.hover', style: {
      'label': 'data(fullTitle)',
      'font-size': '11px',
      'color': '#fff',
      'text-background-color': '#16213e',
      'text-background-opacity': 0.95,
      'text-background-padding': '4px',
      'text-background-shape': 'roundrectangle',
      'text-valign': 'top',
      'text-margin-y': -10,
      'border-color': '#fff',
      'border-width': 3,
      'width': 18, 'height': 18,
      'z-index': 998,
    }},
    // Selected
    { selector: 'node:selected', style: {
      'label': 'data(fullTitle)',
      'font-size': '12px',
      'color': '#e94560',
      'text-background-color': '#16213e',
      'text-background-opacity': 0.95,
      'text-background-padding': '4px',
      'text-background-shape': 'roundrectangle',
      'text-valign': 'top',
      'text-margin-y': -10,
      'border-color': '#e94560',
      'border-width': 4,
      'width': 20, 'height': 20,
      'z-index': 999,
    }},
    // Neighbor highlight
    { selector: 'node.neighbor', style: {
      'label': 'data(label)',
      'font-size': '9px',
      'color': '#e94560',
      'text-background-color': '#16213e',
      'text-background-opacity': 0.9,
      'text-background-padding': '2px',
      'text-background-shape': 'roundrectangle',
      'text-valign': 'bottom',
      'text-margin-y': 5,
      'border-color': '#e94560',
      'border-width': 2,
      'opacity': 1,
    }},
    // Dimmed
    { selector: 'node.dimmed', style: { 'opacity': 0.08 }},
    // --- Edges: hidden by default, visible only on selection ---
    { selector: 'edge', style: {
      'width': 0,
      'opacity': 0,
      'curve-style': 'bezier',
      'target-arrow-shape': 'none',
    }},
    { selector: 'edge.highlight', style: {
      'width': 2,
      'opacity': 0.8,
      'line-color': 'data(color)',
      'target-arrow-color': 'data(color)',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.6,
    }},
    { selector: 'edge.dimmed', style: { 'width': 0, 'opacity': 0 }},
  ],
  layout: { name: 'preset' }, // will run dagre after
  wheelSensitivity: 0.3,
});

// Initial state: hide all documents + edges, show only categories
cy.nodes().not('.concept').hide();
cy.edges().hide();

function layoutVisible() {
  cy.layout({
    name: 'cose',
    fit: true,
    padding: 60,
    nodeRepulsion: 100000,
    idealEdgeLength: 100,
    gravity: 0.3,
    numIter: 200,
    animate: 'end',
    animationDuration: 400,
  }).run();
}
layoutVisible();

const expandedCats = new Set();

// --- Click: category → toggle documents, document → detail ---
cy.on('tap', 'node', function(e) {
  const node = e.target;
  const d = node.data();

  if (d.kind === 'concept') {
    const catTitle = d.fullTitle;
    if (expandedCats.has(catTitle)) {
      // Collapse: hide this category's documents
      expandedCats.delete(catTitle);
      node.neighborhood('node').hide();
      node.connectedEdges().hide();
    } else {
      // Expand: show this category's documents
      expandedCats.add(catTitle);
      node.neighborhood('node').show();
      node.connectedEdges().show();
    }
    // Re-layout visible elements only
    setTimeout(() => layoutVisible(), 100);
    return;
  }

  // Document: highlight neighbors + show sidebar
  cy.elements().removeClass('dimmed neighbor highlight');
  cy.elements(':visible').addClass('dimmed');
  const hood = node.neighborhood().add(node);
  hood.removeClass('dimmed');
  node.neighborhood('node').addClass('neighbor');
  node.connectedEdges(':visible').addClass('highlight');
  node.removeClass('dimmed');
  node.select();
  showDetail(d);
});

cy.on('tap', function(e) {
  if (e.target === cy) {
    cy.elements().removeClass('dimmed neighbor highlight');
    cy.$(':selected').unselect();
  }
});

// Hover
cy.on('mouseover', 'node', function(e) {
  e.target.addClass('hover');
});
cy.on('mouseout', 'node', function(e) {
  e.target.removeClass('hover');
});

// --- Category dropdown ---
const catSelect = document.getElementById('cat-filter');
DATA.categories.forEach(c => {
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  catSelect.appendChild(opt);
});
catSelect.onchange = function() {
  const cat = this.value;
  if (!cat) { cy.elements().show(); relayout(); return; }
  cy.nodes().forEach(n => {
    const d = n.data();
    const match = d.category === cat || (d.tags?.includes('category') && d.fullTitle?.includes(cat));
    if (match) { n.show(); } else { n.hide(); }
  });
  cy.edges().forEach(e => {
    if (e.source().visible() && e.target().visible()) e.show();
    else e.hide();
  });
  relayout();
};

// --- Search ---
document.getElementById('search').oninput = function() {
  const q = this.value.toLowerCase();
  if (!q) { cy.elements().removeClass('dimmed neighbor highlight'); return; }
  const match = cy.nodes().filter(n => (n.data('fullTitle') || '').toLowerCase().includes(q));
  if (match.length > 0) {
    cy.elements().removeClass('dimmed neighbor highlight');
    cy.elements().addClass('dimmed');
    const first = match[0];
    const neighborhood = first.neighborhood().add(first).add(match);
    neighborhood.removeClass('dimmed');
    cy.animate({ center: { eles: first }, zoom: 2 }, { duration: 400 });
    showDetail(first.data());
  }
};

// --- Stats ---
document.getElementById('stats').textContent =
  DATA.nodes.length + ' nodes · ' + DATA.links.length + ' edges · ' + DATA.categories.length + ' categories';

// --- Legend ---
const legendDiv = document.getElementById('legend');
Object.entries(KIND_COLORS).forEach(([k, c]) => {
  if (DATA.kinds.includes(k)) {
    legendDiv.innerHTML += '<div class="legend-item"><div class="legend-dot" style="background:' + c + '"></div>' + k + '</div>';
  }
});

// --- Sidebar detail ---
function showDetail(d) {
  const detail = document.getElementById('detail');
  detail.style.display = 'block';
  document.getElementById('placeholder').style.display = 'none';

  const nodeId = d.id;
  const outEdges = DATA.links.filter(l => l.source === nodeId);
  const inEdges = DATA.links.filter(l => l.target === nodeId);

  let edgesHtml = '';
  inEdges.forEach(e => {
    const src = nodeMap[e.source];
    edgesHtml += '<div class="edge-item" onclick="focusNode(\\'' + e.source + '\\')">← ' + e.kind + ': ' + (src?.title?.substring(0,40) || e.source) + '</div>';
  });
  outEdges.forEach(e => {
    const tgt = nodeMap[e.target];
    edgesHtml += '<div class="edge-item" onclick="focusNode(\\'' + e.target + '\\')">→ ' + e.kind + ': ' + (tgt?.title?.substring(0,40) || e.target) + '</div>';
  });

  let propsHtml = '';
  if (d.properties) {
    Object.entries(d.properties).forEach(([k,v]) => {
      propsHtml += '<div><span style="color:#888">' + k + ':</span> ' + v + '</div>';
    });
  }

  // Chunk content for document nodes
  const docId = d.properties?.doc_id || '';
  const chunks = DOC_CHUNKS[docId] || [];
  let chunksHtml = '';
  if (chunks.length > 0) {
    chunksHtml = '<div class="label">Document Content (' + chunks.length + ' chunks)</div>';
    chunks.forEach(c => {
      chunksHtml += '<div style="margin:6px 0;padding:8px;background:#0f3460;border-radius:4px;border-left:3px solid #3498db">' +
        '<div style="font-size:10px;color:#888;margin-bottom:4px">Chunk #' + c.index + '</div>' +
        '<div style="font-size:12px;line-height:1.5;color:#ccc">' + c.content + '</div></div>';
    });
  }

  detail.innerHTML =
    '<h3 style="color:' + (d.color || '#999') + '">' + (d.kind || '').toUpperCase() + '</h3>' +
    '<div class="label">Title</div><div class="value" style="font-size:15px;font-weight:bold">' + (d.fullTitle || d.label) + '</div>' +
    (d.category ? '<div class="label">Category</div><div class="value">' + d.category + '</div>' : '') +
    (d.tags?.length ? '<div class="label">Tags</div><div class="value">' + d.tags.join(', ') + '</div>' : '') +
    (propsHtml ? '<div class="label">Properties</div><div style="font-size:12px">' + propsHtml + '</div>' : '') +
    '<div class="label">Edges (' + inEdges.length + ' in, ' + outEdges.length + ' out)</div>' +
    '<div class="edge-list">' + (edgesHtml || '<div style="color:#555">No edges</div>') + '</div>' +
    chunksHtml +
    '<div class="label" style="margin-top:12px">Node ID</div>' +
    '<div class="value" style="font-size:11px;color:#555">' + d.id + '</div>';
}

function focusNode(nodeId) {
  const node = cy.getElementById(nodeId);
  if (node.length > 0) {
    cy.elements().removeClass('dimmed neighbor highlight hover');
    cy.elements().addClass('dimmed');
    const neighborhood = node.neighborhood().add(node);
    neighborhood.removeClass('dimmed');
    node.neighborhood('node').addClass('neighbor');
    node.neighborhood('edge').addClass('highlight');
    node.select();
    cy.animate({ center: { eles: node }, zoom: 2 }, { duration: 400 });
    showDetail(node.data());
  }
}
</script>
</body>
</html>"""


async def main() -> int:
    args = _parse_args()

    if not args.graph.exists():
        print(f"ERROR: graph not found: {args.graph}")
        return 1

    print(f"Loading graph from {args.graph}...")
    data = await _load_graph_data(args)
    print(f"  Nodes: {len(data['nodes'])}, Edges: {len(data['links'])}")
    print(f"  Categories: {data['categories']}")
    print(f"  Kinds: {data['kinds']}")

    # Generate HTML
    html_content = (
        HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__KIND_COLORS__", json.dumps(KIND_COLORS))
        .replace("__EDGE_COLORS__", json.dumps(EDGE_COLORS))
        .replace("__DOC_CHUNKS__", json.dumps(data.get("doc_chunks", {}), ensure_ascii=False))
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    print(f"\n✓ Graph explorer → {args.output}")
    print(f"  Open in browser: file://{args.output.resolve()}")

    if not args.no_open:
        webbrowser.open(f"file://{args.output.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
