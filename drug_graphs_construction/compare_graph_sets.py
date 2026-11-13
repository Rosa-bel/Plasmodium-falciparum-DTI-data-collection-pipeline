import os
import argparse
import glob
import torch
import csv


def load_graph(path):
    return torch.load(path , weights_only=False)


def compare_tensors(a, b, atol=1e-6, rtol=1e-5):
    if a.shape != b.shape:
        return False, None
    try:
        close = torch.allclose(a, b, rtol=rtol, atol=atol)
        max_diff = float(torch.max(torch.abs(a - b)).item())
        return bool(close), max_diff
    except Exception:
        return False, None


def compare_graphs(g1, g2, atol=1e-6, rtol=1e-5):
    res = {}

    # Compare node features `x`
    x1 = getattr(g1, 'x', None)
    x2 = getattr(g2, 'x', None)
    if x1 is None and x2 is None:
        res['x_present'] = 'none'
        res['x_equal'] = True
        res['x_max_diff'] = 0.0
    elif (x1 is None) != (x2 is None):
        res['x_present'] = 'mismatch'
        res['x_equal'] = False
        res['x_max_diff'] = None
    else:
        eq, md = compare_tensors(x1, x2, atol=atol, rtol=rtol)
        res['x_present'] = 'both'
        res['x_equal'] = eq
        res['x_max_diff'] = md

    # Compare edge_index
    ei1 = getattr(g1, 'edge_index', None)
    ei2 = getattr(g2, 'edge_index', None)
    if ei1 is None and ei2 is None:
        res['edge_index_equal'] = True
    elif (ei1 is None) != (ei2 is None):
        res['edge_index_equal'] = False
    else:
        res['edge_index_equal'] = bool(torch.equal(ei1, ei2))

    # Compare edge_attr
    ea1 = getattr(g1, 'edge_attr', None)
    ea2 = getattr(g2, 'edge_attr', None)
    if ea1 is None and ea2 is None:
        res['edge_attr_equal'] = True
        res['edge_attr_max_diff'] = 0.0
    elif (ea1 is None) != (ea2 is None):
        res['edge_attr_equal'] = False
        res['edge_attr_max_diff'] = None
    else:
        eq, md = compare_tensors(ea1, ea2, atol=atol, rtol=rtol)
        res['edge_attr_equal'] = eq
        res['edge_attr_max_diff'] = md

    # Quick structural checks
    res['num_nodes_1'] = int(x1.shape[0]) if x1 is not None else 0
    res['num_nodes_2'] = int(x2.shape[0]) if x2 is not None else 0
    res['num_edges_dir_1'] = int(ei1.shape[1]) if ei1 is not None else 0
    res['num_edges_dir_2'] = int(ei2.shape[1]) if ei2 is not None else 0

    return res


def main():
    parser = argparse.ArgumentParser(description='Compare two graph directories of .pt PyG Data objects')
    parser.add_argument('dir_a', help='Directory A (original)')
    parser.add_argument('dir_b', help='Directory B (generated)')
    parser.add_argument('--pattern', default='*.pt', help='Glob pattern to match files')
    parser.add_argument('--out', default='compare_report.txt', help='Output text report')
    parser.add_argument('--csv', default=None, help='Optional CSV summary output')
    parser.add_argument('--atol', type=float, default=1e-6, help='Absolute tolerance for float comparisons')
    parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for float comparisons')
    args = parser.parse_args()

    a_pattern = os.path.join(args.dir_a, args.pattern)
    b_pattern = os.path.join(args.dir_b, args.pattern)
    a_files = glob.glob(a_pattern)
    b_files = glob.glob(b_pattern)

    a_map = {os.path.splitext(os.path.basename(p))[0]: p for p in a_files}
    b_map = {os.path.splitext(os.path.basename(p))[0]: p for p in b_files}

    all_keys = sorted(set(a_map.keys()) | set(b_map.keys()))

    results = []
    for key in all_keys:
        a_path = a_map.get(key)
        b_path = b_map.get(key)
        entry = {'id': key, 'a_path': a_path, 'b_path': b_path}
        if a_path is None:
            entry['status'] = 'missing_in_A'
            results.append(entry)
            continue
        if b_path is None:
            entry['status'] = 'missing_in_B'
            results.append(entry)
            continue

        try:
            g1 = load_graph(a_path)
            g2 = load_graph(b_path)
            comp = compare_graphs(g1, g2, atol=args.atol, rtol=args.rtol)
            entry.update(comp)
            # Determine overall equality
            entry['equal'] = comp.get('x_equal', False) and comp.get('edge_index_equal', False) and comp.get('edge_attr_equal', False)
            entry['status'] = 'equal' if entry['equal'] else 'different'
        except Exception as e:
            entry['status'] = 'error'
            entry['error'] = str(e)

        results.append(entry)

    # Write text report
    with open(args.out, 'w') as fh:
        fh.write(f"Comparison report: {args.dir_a} vs {args.dir_b}\n")
        fh.write(f"Total candidates: {len(all_keys)}\n\n")

        counts = {'equal': 0, 'different': 0, 'missing_in_A': 0, 'missing_in_B': 0, 'error': 0}
        for r in results:
            st = r.get('status')
            counts[st] = counts.get(st, 0) + 1

        fh.write('Summary:\n')
        for k, v in counts.items():
            fh.write(f"  {k}: {v}\n")
        fh.write('\nDetailed per-file results:\n')
        for r in results:
            fh.write(f"- {r['id']}: {r['status']}\n")
            if r['status'] in ('different', 'equal'):
                fh.write(f"    nodes: {r.get('num_nodes_1')} vs {r.get('num_nodes_2')}, ")
                fh.write(f"dir_edges: {r.get('num_edges_dir_1')} vs {r.get('num_edges_dir_2')}\n")
                fh.write(f"    x_equal: {r.get('x_equal')}, x_max_diff: {r.get('x_max_diff')}\n")
                fh.write(f"    edge_index_equal: {r.get('edge_index_equal')}, edge_attr_equal: {r.get('edge_attr_equal')}, edge_attr_max_diff: {r.get('edge_attr_max_diff')}\n")
            if r.get('error'):
                fh.write(f"    ERROR: {r['error']}\n")

    print(f"Wrote report to {args.out}")

    if args.csv:
        keys = ['id', 'status', 'a_path', 'b_path', 'num_nodes_1', 'num_nodes_2', 'num_edges_dir_1', 'num_edges_dir_2', 'x_equal', 'x_max_diff', 'edge_index_equal', 'edge_attr_equal', 'edge_attr_max_diff', 'error']
        with open(args.csv, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            for r in results:
                outrow = {k: r.get(k) for k in keys}
                writer.writerow(outrow)
        print(f"Wrote CSV summary to {args.csv}")


if __name__ == '__main__':
    main()
