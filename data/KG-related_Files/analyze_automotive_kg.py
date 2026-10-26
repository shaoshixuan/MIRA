"""Summarize a KG dataset without changing or copying its source files.

Usage:
    python analyze_automotive_kg.py DATA_DIR --output OUTPUT.json

Triples in this dataset are stored as head, tail, relation (not head, relation,
tail). Counts use the directed graph and do not add inverse relations.
"""

import argparse
import json
from array import array
from collections import Counter
from pathlib import Path


FILE_PATTERNS = {
    "users": ("kg_users*.txt",),
    "items": ("kg_items*.txt",),
    "other_entities": ("kg_other_entities*.txt", "kg_entities*.txt"),
    "relations": ("kg_relations*.txt",),
    "train_triples": ("kg_train_triples*.txt",),
    "other_triples": ("kg_other_triples*.txt",),
}


def rows(path, fields):
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != fields:
                raise ValueError(f"{path}:{line_number}: expected {fields} fields")
            yield parts


def summarize(data_dir):
    paths = {}
    for key, patterns in FILE_PATTERNS.items():
        matches = sorted({path for pattern in patterns for path in data_dir.glob(pattern)})
        if len(matches) != 1:
            raise ValueError(f"expected one {key} file in {data_dir}; found {matches}")
        paths[key] = matches[0]

    entity_types = {}
    counts = Counter()
    type_counts = Counter()
    duplicate_entity_ids = 0
    for group in ("users", "items", "other_entities"):
        for label, raw_id in rows(paths[group], 2):
            entity_id = int(raw_id)
            if entity_id < 0:
                raise ValueError(f"negative entity ID in {paths[group]}")
            entity_type = label.split("::", 1)[0]
            counts[group] += 1
            type_counts[entity_type] += 1
            if entity_id in entity_types:
                duplicate_entity_ids += 1
            entity_types[entity_id] = entity_type

    relations = {}
    duplicate_relation_ids = 0
    for name, raw_id in rows(paths["relations"], 2):
        relation_id = int(raw_id)
        if relation_id in relations:
            duplicate_relation_ids += 1
        relations[relation_id] = name
        counts["relations"] += 1

    if not entity_types or not relations:
        raise ValueError("entity and relation files must not be empty")
    capacity = max(entity_types) + 1
    relation_base = max(relations) + 1
    indegree = array("I", [0]) * capacity
    outdegree = array("I", [0]) * capacity
    parent = array("I", range(capacity))
    size = array("I", [1]) * capacity
    unique_triples = set()
    unique_pairs = set()
    relation_counts = Counter()
    relation_file_counts = {"train_triples": Counter(), "other_triples": Counter()}
    bad_head = bad_tail = bad_relation = self_loops = 0
    bad_samples = []

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(head, tail):
        head, tail = find(head), find(tail)
        if head != tail:
            if size[head] < size[tail]:
                head, tail = tail, head
            parent[tail] = head
            size[head] += size[tail]

    for file_key in ("train_triples", "other_triples"):
        for line_number, (raw_head, raw_tail, raw_relation) in enumerate(rows(paths[file_key], 3), 1):
            head, tail, relation = int(raw_head), int(raw_tail), int(raw_relation)
            counts[file_key] += 1
            relation_counts[relation] += 1
            relation_file_counts[file_key][relation] += 1
            valid_head = head in entity_types
            valid_tail = tail in entity_types
            valid_relation = relation in relations
            bad_head += not valid_head
            bad_tail += not valid_tail
            bad_relation += not valid_relation
            if not (valid_head and valid_tail and valid_relation):
                if len(bad_samples) < 5:
                    bad_samples.append({"file": paths[file_key].name, "line": line_number,
                                        "triple": [head, tail, relation]})
                continue
            outdegree[head] += 1
            indegree[tail] += 1
            self_loops += head == tail
            pair_key = head * capacity + tail
            unique_pairs.add(pair_key)
            unique_triples.add(pair_key * relation_base + relation)
            union(head, tail)

    total_entities = len(entity_types)
    valid_triples = sum(outdegree)
    active_nodes = sum(indegree[i] + outdegree[i] > 0 for i in entity_types)
    head_nodes = {i for i in entity_types if outdegree[i]}
    tail_nodes = sum(bool(indegree[i]) for i in entity_types)
    isolated_by_type = Counter(entity_types[i] for i in entity_types if not (indegree[i] or outdegree[i]))
    component_sizes = Counter()
    for node in entity_types:
        component_sizes[find(node)] += 1
    matrix_pairs = sum(divmod(pair, capacity)[1] in head_nodes for pair in unique_pairs)
    matrix_triples = sum(indegree[node] for node in head_nodes)
    top_relations = [
        {"id": relation_id, "name": relations[relation_id], "triples": relation_counts[relation_id],
         "train": relation_file_counts["train_triples"][relation_id],
         "other": relation_file_counts["other_triples"][relation_id]}
        for relation_id in sorted(relations, key=lambda r: (-relation_counts[r], r))
    ]
    return {
        "source_directory": str(data_dir),
        "definitions": {
            "triple": "one source row (head, tail, relation); directed; no inverse relations added",
            "unique_triple": "distinct (head, tail, relation)",
            "unique_directed_edge": "distinct (head, tail), ignoring relation and duplicate rows",
            "active_node": "entity occurring as head or tail of a valid triple",
            "legacy_matrix": "kg_matrix_build.py indexes only heads and drops triples whose tail is not a head",
        },
        "entities": {
            "user_rows": counts["users"], "item_rows": counts["items"],
            "other_rows": counts["other_entities"], "total_rows": counts["users"] + counts["items"] + counts["other_entities"],
            "unique_ids": total_entities, "duplicate_ids": duplicate_entity_ids,
            "min_id": min(entity_types), "max_id": max(entity_types),
            "id_gaps": capacity - total_entities,
            "by_type": dict(sorted(type_counts.items())),
        },
        "relations": {"rows": counts["relations"], "unique_ids": len(relations),
                      "duplicate_ids": duplicate_relation_ids, "used_ids": sum(bool(relation_counts[r]) for r in relations),
                      "by_frequency": top_relations},
        "triples": {"train_rows": counts["train_triples"], "other_rows": counts["other_triples"],
                    "total_rows": counts["train_triples"] + counts["other_triples"],
                    "valid_rows": valid_triples, "unique": len(unique_triples),
                    "duplicate_valid_rows": valid_triples - len(unique_triples),
                    "self_loop_rows": self_loops},
        "graph": {"directed_edge_rows": valid_triples, "unique_directed_edges": len(unique_pairs),
                  "parallel_edge_rows": valid_triples - len(unique_pairs),
                  "active_nodes": active_nodes, "head_nodes": len(head_nodes), "tail_nodes": tail_nodes,
                  "tail_only_nodes": active_nodes - len(head_nodes),
                  "isolated_nodes": total_entities - active_nodes,
                  "isolated_by_type": dict(sorted(isolated_by_type.items())),
                  "mean_in_degree_all_entities": valid_triples / total_entities,
                  "mean_out_degree_all_entities": valid_triples / total_entities,
                  "max_in_degree": max(indegree), "max_out_degree": max(outdegree),
                  "density_unique_directed_pairs": len(unique_pairs) / (total_entities * total_entities),
                  "weak_components_including_isolates": len(component_sizes),
                  "largest_weak_component_nodes": max(component_sizes.values())},
        "legacy_matrix": {"indexed_nodes": len(head_nodes),
                          "retained_triple_rows": matrix_triples,
                          "dropped_triple_rows_tail_not_indexed": valid_triples - matrix_triples,
                          "nonzero_matrix_entries": matrix_pairs},
        "data_quality": {"unknown_head_rows": bad_head, "unknown_tail_rows": bad_tail,
                         "unknown_relation_rows": bad_relation, "invalid_examples": bad_samples},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.data_dir)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()
