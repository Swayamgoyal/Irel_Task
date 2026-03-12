"""
Steps 7-8: Graph Construction & Output

Builds a directed prerequisite graph from the detected dependencies.

Outputs:
  1. JSON — Machine-readable complete output
  2. Interactive HTML graph — Visual prerequisite map (pyvis)
  3. PNG graph — Static visualization (matplotlib)
  4. DOT format — For external tools (Graphviz)
"""
import json
from pathlib import Path

import networkx as nx

from pipeline.config import get_video_dir


def build_graph(concepts_data: dict, prerequisites_data: dict) -> nx.DiGraph:
    """
    Build a directed graph of concept prerequisites.

    Nodes = concepts, Edges = prerequisite relationships.
    Edge direction: prerequisite → dependent (A → B means "learn A before B").

    Args:
        concepts_data: Output from Step 5.
        prerequisites_data: Output from Step 6.

    Returns:
        NetworkX DiGraph.
    """
    G = nx.DiGraph()

    # Add concept nodes with attributes
    for concept in concepts_data.get("concepts", []):
        G.add_node(
            concept["id"],
            label=concept["name"],
            type=concept.get("type", "unknown"),
            description=concept.get("description", ""),
            aliases=concept.get("aliases", []),
        )

    # Build a lookup: any token that starts with a known concept_id → canonical id
    # This handles LLM returning "concept_3 (Force)" instead of "concept_3"
    canonical = {}
    for node_id in G.nodes():
        canonical[node_id] = node_id
        canonical[node_id.lower()] = node_id
    # Also map by concept name (case-insensitive)
    for concept in concepts_data.get("concepts", []):
        canonical[concept["name"].lower()] = concept["id"]

    def _resolve(raw: str) -> str | None:
        """Resolve a raw dependency string to a canonical concept ID."""
        raw = raw.strip()
        # Direct match
        if raw in canonical:
            return canonical[raw]
        # Strip trailing " (Name)" suffix: "concept_3 (Force)" → "concept_3"
        base = raw.split("(")[0].strip()
        if base in canonical:
            return canonical[base]
        # Case-insensitive
        if raw.lower() in canonical:
            return canonical[raw.lower()]
        if base.lower() in canonical:
            return canonical[base.lower()]
        return None

    # Add dependency edges
    skipped = 0
    for dep in prerequisites_data.get("dependencies", []):
        from_id = _resolve(dep["from_concept"])
        to_id = _resolve(dep["to_concept"])
        if from_id and to_id:
            G.add_edge(
                from_id,
                to_id,
                relationship=dep.get("relationship_type", "PREREQUISITE"),
                strength=dep.get("strength", 0.5),
                justification=dep.get("justification", ""),
            )
        else:
            skipped += 1
            print(f"[Step 7] WARNING: could not resolve edge "
                  f"'{dep['from_concept']}' → '{dep['to_concept']}' — skipped")

    print(f"[Step 7] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
          + (f" ({skipped} skipped)" if skipped else ""))

    # Check for cycles (prerequisite graphs should be DAGs)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        print(f"[Step 7] WARNING: {len(cycles)} cycles detected. Removing weakest edges...")
        for cycle in cycles:
            # Remove the weakest edge in each cycle
            min_strength = float("inf")
            min_edge = None
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    s = G[u][v].get("strength", 0.5)
                    if s < min_strength:
                        min_strength = s
                        min_edge = (u, v)
            if min_edge and G.has_edge(*min_edge):
                G.remove_edge(*min_edge)
                print(f"[Step 7] Removed cycle edge: {min_edge}")

    return G


def _type_color(concept_type: str) -> str:
    """Map concept type to a color."""
    return {
        "core": "#e74c3c",         # red
        "supporting": "#3498db",   # blue
        "prerequisite": "#2ecc71", # green
    }.get(concept_type, "#95a5a6") # grey


def generate_interactive_html(G: nx.DiGraph, output_path: Path | None = None, video_id: str = "", title: str = "") -> Path:
    """
    Generate a fully self-contained interactive HTML graph visualization.

    Embeds vis-network inline so no CDN or external files are needed.
    Works in any browser, VS Code webview, or offline.
    """
    if output_path is None:
        if video_id:
            output_path = get_video_dir(video_id) / "prerequisite_graph.html"
        else:
            output_path = Path("prerequisite_graph.html")

    # Load vis-network JS (embedded inline)
    vis_js_path = Path(__file__).parent / "assets" / "vis-network.min.js"
    vis_js = vis_js_path.read_text(encoding="utf-8")

    # Build nodes/edges JSON
    import json as _json

    is_dag = nx.is_directed_acyclic_graph(G)
    page_title = title if title else "Prerequisite Graph"

    nodes_list = []
    for node_id, attrs in G.nodes(data=True):
        concept_type = attrs.get("type", "unknown")
        color = _type_color(concept_type)
        out_degree = G.out_degree(node_id)
        in_degree = G.in_degree(node_id)
        aliases = attrs.get("aliases", [])
        alias_str = ", ".join(aliases) if aliases else "—"
        nodes_list.append({
            "id": node_id,
            "label": attrs.get("label", node_id),
            "color": {"background": color, "border": "#ffffff", "highlight": {"background": color, "border": "#fff"}},
            "size": 22 + in_degree * 4 + out_degree * 4,
            "shape": "dot",
            "title": (
                f"<div style='max-width:280px'>"
                f"<b style='font-size:14px'>{attrs.get('label', node_id)}</b><br>"
                f"<span style='color:#aaa'>Type: {concept_type}</span><br>"
                f"<span style='color:#aaa'>Also known as: {alias_str}</span><br><br>"
                f"Prerequisites: {in_degree} &nbsp;|&nbsp; Dependents: {out_degree}<br><br>"
                f"{attrs.get('description', '')[:250]}"
                f"</div>"
            ),
            "level": None,
        })

    relationship_colors = {
        "HARD_PREREQUISITE": "#e74c3c",
        "SOFT_PREREQUISITE": "#f39c12",
        "BUILDS_UPON": "#3498db",
        "USES": "#2ecc71",
    }

    edges_list = []
    for u, v, attrs in G.edges(data=True):
        rel_type = attrs.get("relationship", "PREREQUISITE")
        strength = attrs.get("strength", 0.5)
        edges_list.append({
            "from": u,
            "to": v,
            "color": relationship_colors.get(rel_type, "#cccccc"),
            "width": 1 + strength * 3,
            "title": (
                f"{rel_type}<br>"
                f"Strength: {strength}<br>"
                f"{attrs.get('justification', '')[:150]}"
            ),
            "arrows": "to",
        })

    nodes_json = _json.dumps(nodes_list)
    edges_json = _json.dumps(edges_list)

    layout_options = """
    layout: {
      hierarchical: {
        enabled: true,
        direction: "LR",
        sortMethod: "directed",
        levelSeparation: 220,
        nodeSpacing: 130,
        treeSpacing: 200,
        blockShifting: true,
        edgeMinimization: true,
        parentCentralization: true
      }
    },
    physics: { enabled: false },""" if is_dag else """
    physics: {
      forceAtlas2Based: { gravitationalConstant: -120, centralGravity: 0.01,
                          springLength: 220, springConstant: 0.08 },
      solver: "forceAtlas2Based",
      stabilization: { iterations: 300 }
    },"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{page_title} — Prerequisite Graph</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: #0f0f23; font-family: 'Segoe UI', Arial, sans-serif; }}
  #header {{ position: absolute; top: 0; left: 0; right: 0; z-index: 20;
             background: linear-gradient(135deg, #1a1a3e 0%, #0f0f23 100%);
             padding: 10px 16px; border-bottom: 1px solid #333; }}
  #header h1 {{ margin: 0; color: #fff; font-size: 16px; font-weight: 600; }}
  #header span {{ color: #888; font-size: 12px; }}
  #graph {{ width: 100%; height: 100vh; padding-top: 50px; }}
  #legend {{ position: absolute; top: 60px; right: 12px; background: rgba(15,15,35,0.92);
             padding: 12px 16px; border-radius: 10px; color: #ddd; font-size: 12px;
             z-index: 10; border: 1px solid #333; min-width: 170px; }}
  .legend-section {{ font-weight: 600; color: #aaa; margin: 8px 0 4px; font-size: 11px;
                     text-transform: uppercase; letter-spacing: 0.5px; }}
  .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; margin-right: 8px;
                 flex-shrink: 0; display: inline-block; }}
  .legend-line {{ width: 20px; height: 3px; margin-right: 6px; flex-shrink: 0;
                  display: inline-block; border-radius: 2px; }}
  #stats {{ position: absolute; top: 60px; left: 12px; background: rgba(15,15,35,0.92);
            padding: 10px 14px; border-radius: 10px; color: #ddd; font-size: 12px;
            z-index: 10; border: 1px solid #333; }}
  #stats div {{ margin: 2px 0; }}
  #stats .val {{ color: #7ec8e3; font-weight: 600; }}
</style>
</head>
<body>
<div id="header">
  <h1>{page_title} &mdash; Prerequisite Dependency Graph</h1>
  <span>Hover nodes/edges for details &bull; Drag to rearrange &bull; Scroll to zoom</span>
</div>
<div id="stats">
  <div>Concepts: <span class="val">{G.number_of_nodes()}</span></div>
  <div>Dependencies: <span class="val">{G.number_of_edges()}</span></div>
  <div>Structure: <span class="val">{'DAG' if is_dag else 'Cyclic'}</span></div>
</div>
<div id="legend">
  <div class="legend-section">Concept Types</div>
  <div class="legend-item"><span class="legend-dot" style="background:#e74c3c"></span>Core</div>
  <div class="legend-item"><span class="legend-dot" style="background:#3498db"></span>Supporting</div>
  <div class="legend-item"><span class="legend-dot" style="background:#2ecc71"></span>Prerequisite</div>
  <div class="legend-section">Edge Types</div>
  <div class="legend-item"><span class="legend-line" style="background:#e74c3c"></span>Hard Prereq</div>
  <div class="legend-item"><span class="legend-line" style="background:#f39c12"></span>Soft Prereq</div>
  <div class="legend-item"><span class="legend-line" style="background:#3498db"></span>Builds Upon</div>
  <div class="legend-item"><span class="legend-line" style="background:#2ecc71"></span>Uses</div>
</div>
<div id="graph"></div>
<script>{vis_js}</script>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var container = document.getElementById("graph");
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  {layout_options}
  edges: {{
    arrows: {{ to: {{ enabled: true, scaleFactor: 1.0 }} }},
    smooth: {{ type: "cubicBezier", forceDirection: "horizontal", roundness: 0.4 }},
    font: {{ size: 10, color: "#aaa", align: "middle" }}
  }},
  nodes: {{
    font: {{ size: 15, face: "Segoe UI", color: "#ffffff", strokeWidth: 3, strokeColor: "#0f0f23" }},
    borderWidth: 2,
    borderWidthSelected: 3,
    shadow: {{ enabled: true, color: "rgba(0,0,0,0.5)", size: 8, x: 2, y: 2 }}
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 80,
    navigationButtons: true,
    keyboard: true
  }}
}};
new vis.Network(container, data, options);
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[Step 8] Interactive graph saved to: {output_path}")
    return output_path


def generate_static_png(G: nx.DiGraph, output_path: Path | None = None, video_id: str = "") -> Path:
    """Generate a static PNG visualization using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if output_path is None:
        if video_id:
            output_path = get_video_dir(video_id) / "prerequisite_graph.png"
        else:
            output_path = Path("prerequisite_graph.png")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Layout
    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # Node colors by type
    node_colors = [_type_color(G.nodes[n].get("type", "unknown")) for n in G.nodes()]
    node_sizes = [300 + G.out_degree(n) * 150 for n in G.nodes()]

    # Draw
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#666666",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        width=1.5,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=2,
    )

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_weight="bold",
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#e74c3c", label="Core Concept"),
        mpatches.Patch(color="#3498db", label="Supporting"),
        mpatches.Patch(color="#2ecc71", label="Prerequisite"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=10)
    ax.set_title("Prerequisite Dependency Graph", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Step 8] Static graph saved to: {output_path}")
    return output_path


def generate_final_output(
    transcript_data: dict,
    language_profile: dict,
    normalized_data: dict,
    concepts_data: dict,
    prerequisites_data: dict,
    graph: nx.DiGraph,
    video_source: str = "",
    video_id: str = "",
) -> dict:
    """
    Generate the final machine-readable JSON output combining all pipeline results.

    Returns:
        Complete output dict.
    """
    # Graph metrics
    metrics = {
        "total_concepts": graph.number_of_nodes(),
        "total_dependencies": graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "graph_density": round(nx.density(graph), 4),
    }

    if nx.is_directed_acyclic_graph(graph):
        metrics["longest_path_length"] = nx.dag_longest_path_length(graph)
        metrics["longest_path"] = nx.dag_longest_path(graph)
        metrics["topological_order"] = list(nx.topological_sort(graph))

    output = {
        "metadata": {
            "video_source": video_source,
            "transcription_model": transcript_data.get("model", "unknown"),
            "detected_language": transcript_data.get("language", "unknown"),
            "language_probability": transcript_data.get("language_probability", 0),
            "audio_duration_seconds": transcript_data.get("duration_seconds", 0),
            "languages_detected": language_profile.get("languages_detected", []),
            "code_mix_ratio": language_profile.get("code_mix_ratio", 0),
        },
        "lecture_info": {
            "topic": concepts_data.get("lecture_topic", ""),
            "summary": concepts_data.get("lecture_summary", ""),
            "teaching_flow": concepts_data.get("teaching_flow", []),
            "analogies_used": concepts_data.get("analogies_used", []),
        },
        "concepts": concepts_data.get("concepts", []),
        "prerequisites": {
            "dependencies": prerequisites_data.get("dependencies", []),
            "learning_path": prerequisites_data.get("learning_path", []),
            "concept_clusters": prerequisites_data.get("concept_clusters", []),
            "root_concepts": prerequisites_data.get("root_concepts", []),
            "leaf_concepts": prerequisites_data.get("leaf_concepts", []),
        },
        "linguistic_analysis": {
            "original_language_distribution": language_profile.get("overall_distribution", {}),
            "technical_terms_mapped": normalized_data.get("technical_terms_mapped", {}),
            "topic_shifts": normalized_data.get("topic_shifts", []),
        },
        "graph_metrics": metrics,
        "normalized_transcript": normalized_data.get("normalized_text", ""),
    }

    # Save final output
    audio_stem = Path(transcript_data.get("audio_file", "output")).stem
    if video_id:
        output_path = get_video_dir(video_id) / f"{audio_stem}_final_output.json"
    else:
        output_path = Path(f"{audio_stem}_final_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Step 8] Final output saved to: {output_path}")
    return output
