#!/usr/bin/env python3
"""Plot one HCZ VTU frame (pass all MPI pieces) with its actual subcell mask.

Example: python3 plotHczSubcells.py 'hcz_rt_sark4_*_0010.vtu' -o interface.png
Requires NumPy and Matplotlib. Reads the solver's ASCII VTU output.
"""
import argparse
import glob
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def read_piece(path):
    root = ET.parse(path)
    piece = root.find('.//Piece')
    if piece is None:
        raise ValueError(f'{path}: missing VTU Piece')

    def array(node, dtype=float):
        if node is None:
            raise ValueError(f'{path}: missing array; rerun with subcell output enabled in the solver')
        return np.fromstring(node.text or '', sep=' ', dtype=dtype)

    xyz = array(piece.find('./Points/DataArray')).reshape(-1, 3)
    cells = array(piece.find('./Cells/DataArray[@Name="connectivity"]'), int).reshape(-1, 4)
    mask = array(piece.find('./CellData/DataArray[@Name="SubcellActive"]'), int)
    ids = array(piece.find('./CellData/DataArray[@Name="ElementId"]'), int)
    seed_node = piece.find('./CellData/DataArray[@Name="InterfaceMarked"]')
    seed = array(seed_node, int) if seed_node is not None else None
    if seed is not None and (len(seed) != len(mask) or not np.all((seed == 0) | (seed == 1))):
        raise ValueError(f'{path}: inconsistent InterfaceMarked array')
    phi = array(piece.find('./PointData/DataArray[@Name="Index"]'))
    pressure = array(piece.find('./PointData/DataArray[@Name="Pressure"]'))
    velocity = array(piece.find('./PointData/DataArray[@Name="Velocity"]')).reshape(-1, 3)
    stamp = array(root.find('.//FieldData/DataArray[@Name="TimeValue"]'))
    if len(stamp) != 1 or len(mask) != len(cells) or len(ids) != len(cells):
        raise ValueError(f'{path}: inconsistent time/cell arrays')
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f'{path}: SubcellActive must contain only 0 and 1')
    return dict(xyz=xyz, cells=cells, mask=mask, interface=seed, ids=ids, phi=phi,
                pressure=pressure, speed=np.linalg.norm(velocity, axis=1), time=stamp[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='VTU files or quoted glob for ONE output frame')
    parser.add_argument('-o', '--output', type=Path, default=Path('hcz_subcells.png'))
    parser.add_argument('--interface-value', type=float,
                        help='Index contour value (default: midpoint of observed index range)')
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        parser.error('no VTU files matched')
    try:
        pieces = [read_piece(p) for p in paths]
    except (ValueError, ET.ParseError) as error:
        parser.error(str(error))
    stamp = pieces[0]['time']
    if any(abs(p['time'] - stamp) > 1e-12 * max(1, abs(stamp)) for p in pieces):
        parser.error('select only one time frame, including its MPI pieces')
    nonempty = [p for p in pieces if len(p['cells'])]
    if not nonempty:
        parser.error('the selected frame has no cells')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.tri import Triangulation
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    limits = {key: (min(p[key].min() for p in nonempty), max(p[key].max() for p in nonempty))
              for key in ('phi', 'pressure', 'speed')}
    level = args.interface_value
    if level is None:
        level = .5 * sum(limits['phi'])
    show_seed = all(p['interface'] is not None for p in nonempty)
    fig, axes = plt.subplots(1, 5 if show_seed else 4, figsize=(15 if show_seed else 12, 8), constrained_layout=True)
    binary = ListedColormap(['#eeeeee', '#ef8a26'])
    binary_norm = BoundaryNorm([-.5, .5, 1.5], 2)
    norms = {key: Normalize(lo, hi if hi > lo else lo + 1e-12) for key, (lo, hi) in limits.items()}
    seen, active, seeds = set(), set(), set()
    for p in nonempty:
        xyz, cells = p['xyz'], p['cells']
        unique = set(p['ids'].tolist())
        if seen & unique:
            parser.error('duplicate elements: select each MPI piece of one run only once')
        seen.update(unique)
        active.update(p['ids'][p['mask'] == 1].tolist())
        if show_seed:
            seeds.update(p['ids'][p['interface'] == 1].tolist())
        triangles = np.concatenate((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]))
        tri = Triangulation(xyz[:, 0], xyz[:, 1], triangles)
        borders, parent_cells, parent_mask, parent_seed = [], [], [], []
        for gid in np.unique(p['ids']):
            vertices = xyz[cells[p['ids'] == gid].ravel(), :2]
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            border = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]),
                      (lo[0], hi[1]), (lo[0], lo[1])]
            flag = int(p['mask'][p['ids'] == gid][0])
            parent_cells.append(border)
            parent_mask.append(flag)
            if show_seed:
                parent_seed.append(int(p['interface'][p['ids'] == gid][0]))
            if flag:
                borders.append(border)
        for ax, key in zip(axes[:3], ('phi', 'pressure', 'speed')):
            ax.tripcolor(tri, p[key], shading='gouraud', cmap='viridis', norm=norms[key])
            ax.add_collection(LineCollection(borders, colors='#ef8a26', linewidths=.7))
        axes[3].add_collection(PolyCollection(parent_cells, array=np.asarray(parent_mask),
                                             cmap=binary, norm=binary_norm,
                                             edgecolors='#cccccc', linewidths=.15))
        if show_seed:
            axes[4].add_collection(PolyCollection(parent_cells, array=np.asarray(parent_seed),
                                                  cmap=binary, norm=binary_norm,
                                                  edgecolors='#cccccc', linewidths=.15))
        for ax in axes:
            if p['phi'].min() < level < p['phi'].max():
                ax.tricontour(tri, p['phi'], levels=[level], colors='black', linewidths=.8)
    for ax, key, title in zip(axes[:3], ('phi', 'pressure', 'speed'), ('Index', 'Pressure', 'Speed')):
        bar = fig.colorbar(ScalarMappable(norm=norms[key], cmap='viridis'), ax=ax,
                           orientation='horizontal', pad=.04)
        bar.locator = MaxNLocator(nbins=4)
        bar.formatter = ScalarFormatter(useMathText=True)
        bar.formatter.set_powerlimits((-3, 3))
        bar.update_ticks()
        ax.set_title(title)
    axes[3].set_title(f'Subcell mask: {len(active)}/{len(seen)} elements')
    bar = fig.colorbar(ScalarMappable(norm=binary_norm, cmap=binary), ax=axes[3],
                       ticks=[0, 1], orientation='horizontal', pad=.04)
    bar.ax.set_xticklabels(['DG', 'WENO + buffer'])
    if show_seed:
        axes[4].set_title(f'Interface seed: {len(seeds)}/{len(seen)} elements')
        bar = fig.colorbar(ScalarMappable(norm=binary_norm, cmap=binary), ax=axes[4],
                           ticks=[0, 1], orientation='horizontal', pad=.04)
        bar.ax.set_xticklabels(['Unmarked', 'Interface'])
    for ax in axes:
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.suptitle(f'HCZ, t = {stamp:g}\nOrange: subcell elements; black: index = {level:g}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f'Saved {args.output} ({len(active)}/{len(seen)} marked elements, t={stamp:g})')


if __name__ == '__main__':
    main()
