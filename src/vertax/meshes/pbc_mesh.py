"""Periodic Boundary Condition on a mesh, useful for bulk tissue dynamics with no explicit boundaries."""

from pathlib import Path
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from vertax.geo import get_area, get_length, get_length_with_offset, get_perimeter, update_pbc
from vertax.mask_analysis import find_vertices_edges_faces, mask_from_image, pad
from vertax.meshes.mesh import Mesh

__all__ = ["PbcMesh"]


class PbcMesh(Mesh):
    """Periodic Boundary Condition on a mesh.

    For a PbcMesh, `vertices` is a 2D array of floats of size (nb_vertices, 2) ;
    with the coordinates of the vertices (in ]0, width[ x ]0, height[ ).

    `edges` is a  2D array of integers of size (nb_half_edges, 8), with:

    - id of previous half-edge,
    - id of next half-edge,
    - id of twin half-edge,
    - id of source vertex,
    - id of target vertex,
    - id of the face containing the half-edge.
    - x-offset (either -1, 0 or 1) depending on the target vertex crossing a boundary.
    - y-offset (either -1, 0 or 1) depending on the target vertex crossing a boundary.

    `faces` is a 1D array of integers of size (nb_faces) containing the id of a half-edge belonging to this face.
    """

    def __init__(self) -> None:
        """Do not call the constructor directly, use the dedicated class methods such as:

        - `from_random_seeds`,
        - `from_seeds`,
        - `from_image`,
        - `from_mask`,
        - `create_empty`.
        """  # noqa: D415
        super().__init__()

        self.MAX_EDGES_IN_ANY_FACE: int = 20
        """Optimization parameter : must be more than the estimated maximum number of edges in a face. Base value is 20."""  # noqa: E501

    @classmethod
    def from_random_seeds(cls, nb_seeds: int, width: float, height: float, random_key: int) -> Self:
        """Create a Periodic Boundary Conditions Mesh from a Voronoi diagram based on random seeds.

        Args:
            nb_seeds (int): Number of random seeds to use.
            width (float): Width of the rectangular domains the seeds will be in.
            height (float): Height of the rectangular domains the seeds will be in.
            random_key (int): Set the random key for reproducibility.

        Returns:
            Self: The corresponding mesh.
        """
        key = jax.random.PRNGKey(random_key)
        seeds = jnp.array((width, height)) * jax.random.uniform(key, (nb_seeds, 2))
        try:
            return cls.from_seeds(seeds, width, height)
        except _FaceOrderingError as exc:
            # `_make_he_structure` cannot name the seed, and the seed is what a caller varies.
            msg = f"{exc} Reached from from_random_seeds(nb_seeds={nb_seeds}, random_key={random_key})."
            raise _FaceOrderingError(msg) from exc

    @classmethod
    def from_seeds(cls, seeds: Array, width: float, height: float) -> Self:
        """Create a Periodic Boundary Conditions Mesh from a Voronoi diagram based on a list of seeds.

        The seeds are assumed to have positive x and y positions.

        Args:
            seeds (Array[float32]): jax float array of seed positions of shape (nbSeeds, 2).
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
        """
        (
            periodic_voronoi_vertices_idx,
            periodic_voronoi_vertices_pos,
            periodic_voronoi_edges,
            offsets,
            periodic_voronoi_faces,
        ) = _make_periodic(seeds, width, height)

        vertices, edges, faces = _make_he_structure(
            width,
            height,
            periodic_voronoi_vertices_idx,
            periodic_voronoi_vertices_pos,
            periodic_voronoi_edges,
            offsets,
            periodic_voronoi_faces,
        )

        pbc_mesh = cls._create()
        pbc_mesh.vertices = jnp.array(vertices, dtype=np.float32)
        pbc_mesh.edges = jnp.array(edges, dtype=np.int32)
        pbc_mesh.faces = jnp.array(faces, dtype=np.int32)
        pbc_mesh.width = width
        pbc_mesh.height = height

        return pbc_mesh

    @classmethod
    def from_image(
        cls,
        image: NDArray,
    ) -> Self:
        """Create a rudimentary mesh with periodic boundary conditions from an image.

        To do that, we perform a segmentation using Cellpose and we try to fill the holes.
        The result will probably be imperfect and it will always be better if you
        provide directly a mask (with no holes) with the function "periodic_from_mask".

        The mask is made periodic by mirroring its edges.

        Args:
            image (NDArray): The image which will act as a template for the mesh.

        Returns:
            Self: The corresponding mesh.
        
        .. warning::

            **Junction extraction is unreliable on real masks, and this method is currently
            unsupported for them.** It builds the mesh with
            `mask_analysis.find_vertices_edges_faces`, which keeps a label triple only where
            exactly one 2x2 window carries it. A real trijunction is generally a small cluster of
            windows, so genuine junctions are discarded and spurious short cycles close in their
            place.

            MEASURED on `examples/image.tif`, the image shipped with this method's own example:
            `V - E + F = 55` and **472 faces from 64 segmented cells**. On a raw microscopy crop,
            `V - E + F = 6`. A correct planar mesh gives 0.

            For a segmented tissue, use :mod:`vertax.trace` instead, which extracts junctions on the
            pixel-corner lattice and is tested against a stored reference.

            **The gap is real and `vertax.trace` does not close it.** `trace` produces a *clamped
            patch* -- a finite tissue fragment with a boundary -- not a periodic mesh. There is
            currently **no working route from an image to a periodic mesh**. Substituting the
            corner-lattice extraction here is the intended fix; it is deferred, and its three
            preconditions are recorded in the freeze checklist.

        """
        return cls.from_mask(mask_from_image(image))

    @classmethod
    def from_mask(  # noqa: C901
        cls,
        mask: NDArray,
    ) -> Self:
        """Create a rudimentary mesh with periodic boundary conditions from a mask with no holes.

        The mask is made periodic by mirroring its edges.

        Args:
            mask (NDArray): The mask with no holes which will act as a template for the mesh.

        Returns:
            Self: The corresponding mesh.
        
        .. warning::

            **Junction extraction is unreliable on real masks, and this method is currently
            unsupported for them.** It builds the mesh with
            `mask_analysis.find_vertices_edges_faces`, which keeps a label triple only where
            exactly one 2x2 window carries it. A real trijunction is generally a small cluster of
            windows, so genuine junctions are discarded and spurious short cycles close in their
            place.

            MEASURED on `examples/image.tif`, the image shipped with this method's own example:
            `V - E + F = 55` and **472 faces from 64 segmented cells**. On a raw microscopy crop,
            `V - E + F = 6`. A correct planar mesh gives 0.

            For a segmented tissue, use :mod:`vertax.trace` instead, which extracts junctions on the
            pixel-corner lattice and is tested against a stored reference.

            **The gap is real and `vertax.trace` does not close it.** `trace` produces a *clamped
            patch* -- a finite tissue fragment with a boundary -- not a periodic mesh. There is
            currently **no working route from an image to a periodic mesh**. Substituting the
            corner-lattice extraction here is the intended fix; it is deferred, and its three
            preconditions are recorded in the freeze checklist.

        """
        padded_mask = pad(mask, save=False, output_path="refined_and_padded_image.tiff")
        # Find vertices, edges, faces
        vertices, edges, faces = find_vertices_edges_faces(padded_mask)

        # imread tiff = Y is the first axis, X the second.
        height: int = mask.shape[0]  # original image length. Padded is 3 times bigger.
        y_min = height / 2
        y_max = 2 * height + (height / 2)
        width: int = mask.shape[1]
        x_min = width / 2
        x_max = 2 * width + (width / 2)

        col0_mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] < x_max)
        col1_mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] < y_max)

        periodic_vertices_idx = np.arange(len(vertices))[col0_mask & col1_mask]
        periodic_vertices_pos = vertices[col0_mask & col1_mask]

        # store map between vertex id -> inside vertex id
        inside_vertex: dict[int, int] = {idx: idx for idx in periodic_vertices_idx}
        for i, vertex in enumerate(vertices):
            if i not in periodic_vertices_idx:
                x, y = vertex
                if x < x_min:
                    x += 2 * width
                elif x >= x_max:
                    x -= 2 * width

                if y < y_min:
                    y += 2 * height
                elif y >= y_max:
                    y -= 2 * height

                # Find corresponding inside vertex to the outside dest vertex
                for idx, pos in zip(periodic_vertices_idx, periodic_vertices_pos, strict=True):
                    if np.max(np.abs(pos - [x, y])) < 1:
                        inside_vertex[i] = idx
                        break

        edges_inside = []
        edges_outside = []
        offsets_inside = {}
        offsets_outside = {}
        visited = []

        for e in edges:
            if e[0] in periodic_vertices_idx and e[1] in periodic_vertices_idx:
                edges_inside.append(tuple(sorted((e[0], e[1]))))
                offsets_inside[(e[0], e[1])] = (0, 0)
                offsets_inside[(e[1], e[0])] = (0, 0)
            elif bool(e[0] in periodic_vertices_idx) != bool(e[1] in periodic_vertices_idx):
                if e[0] in periodic_vertices_idx:
                    # origin in, dest out
                    # check x coord
                    if vertices[e[1]][0] < x_min:
                        offset_x1 = -1
                    elif vertices[e[1]][0] >= x_max:
                        offset_x1 = 1
                    else:
                        offset_x1 = 0

                    # Now check y coord
                    if vertices[e[1]][1] < y_min:
                        offset_y1 = -1
                    elif vertices[e[1]][1] >= y_max:
                        offset_y1 = 1
                    else:
                        offset_y1 = 0

                    # Find corresponding inside vertex to the outside dest vertex
                    if e[1] not in inside_vertex:
                        print(f"Error, no inside vertex found for vertex {e[1]}.")
                    else:
                        idx = inside_vertex[e[1]]
                        edges_outside.append(tuple(sorted((e[0], idx))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                            offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                else:
                    # dest in, origin out
                    if vertices[e[0]][0] < x_min:
                        offset_x0 = -1
                    elif vertices[e[0]][0] >= x_max:
                        offset_x0 = 1
                    else:
                        offset_x0 = 0

                    if vertices[e[0]][1] < y_min:
                        offset_y0 = -1
                    elif vertices[e[0]][1] >= y_max:
                        offset_y0 = 1
                    else:
                        offset_y0 = 0

                    # Find corresponding inside vertex to the outside dest vertex
                    if e[0] not in inside_vertex:
                        print(f"Error, no inside vertex found for vertex {e[0]}.")
                    else:
                        idx = inside_vertex[e[0]]
                        edges_outside.append(tuple(sorted((idx, e[1]))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(idx, e[1])] = (-offset_x0, -offset_y0)
                            offsets_outside[(e[1], idx)] = (offset_x0, offset_y0)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))

        periodic_edges = list(set(edges_inside)) + list(set(edges_outside))
        offsets = offsets_inside | offsets_outside

        periodic_faces: list[set[int]] = [
            {inside_vertex[i] for i in face} for face in faces if any(v_id in periodic_vertices_idx for v_id in face)
        ]

        vertices, edges, faces = _make_he_structure(
            2 * width,
            2 * height,
            periodic_vertices_idx,  # ty:ignore[invalid-argument-type]
            periodic_vertices_pos,
            periodic_edges,
            offsets,
            periodic_faces,
            vertices_offset=(x_min, y_min),
        )

        pbc_mesh = cls._create()
        pbc_mesh.vertices = jnp.array(vertices, dtype=np.float32)
        pbc_mesh.edges = jnp.array(edges, dtype=np.int32)
        pbc_mesh.faces = jnp.array(faces, dtype=np.int32)
        pbc_mesh.width = 2 * width
        pbc_mesh.height = 2 * height

        return pbc_mesh

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty mesh. Use if you know what you're doing !"""
        return cls._create()

    @classmethod
    def copy_mesh(cls, other_mesh: Self) -> Self:
        """Copy all parameters from another mesh in a new mesh."""
        mesh = cls._create()
        mesh.vertices = other_mesh.vertices.copy()
        mesh.edges = other_mesh.edges.copy()
        mesh.faces = other_mesh.faces.copy()
        mesh.width = other_mesh.width
        mesh.height = other_mesh.height
        mesh.vertices_params = other_mesh.vertices_params.copy()
        mesh.edges_params = other_mesh.edges_params.copy()
        mesh.faces_params = other_mesh.faces_params.copy()
        mesh.MAX_EDGES_IN_ANY_FACE = other_mesh.MAX_EDGES_IN_ANY_FACE

        return mesh

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        All PBCMesh data is saved.

        Args:
            path (str): Path to the saved file. The extension is .npz, a numpy format.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            allow_pickle=False,
            vertices=self.vertices,
            edges=self.edges,
            faces=self.faces,
            width=self.width,
            height=self.height,
            vertices_params=self.vertices_params,
            edges_params=self.edges_params,
            faces_params=self.faces_params,
            MAX_EDGES_IN_ANY_FACE=self.MAX_EDGES_IN_ANY_FACE,
        )

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        All PBCMesh data is reloaded.

        Args:
            path (str): Path to the mesh file (.npz), numpy format.

        Returns:
            Mesh: the mesh loaded from the numpy .npz file.
        """
        mesh_file = np.load(path)
        mesh = cls._create()
        mesh.vertices, mesh.edges, mesh.faces = (
            jnp.array(mesh_file["vertices"]),
            jnp.array(mesh_file["edges"].reshape(-1, 8)),
            jnp.array(mesh_file["faces"]),
        )
        mesh.width, mesh.height = float(mesh_file["width"]), float(mesh_file["height"])
        mesh.vertices_params = jnp.array(mesh_file["vertices_params"])
        mesh.edges_params = jnp.array(mesh_file["edges_params"])
        mesh.faces_params = jnp.array(mesh_file["faces_params"])
        mesh.MAX_EDGES_IN_ANY_FACE = int(mesh_file["MAX_EDGES_IN_ANY_FACE"])
        return mesh

    def save_mesh_txt(
        self,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> None:
        """Save a mesh in separate text files that can be read by numpy.

        Only save the vertices, edges and faces, not other parameters.

        Args:
            directory (str): Path to the directory where to save the files.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
            vertices_params_filename (str, optional): Filename for the vertices parameters table.
                    Defaults to "vertParamsTable.txt".
            edges_params_filename (str, optional): Filename for the half-edges parameters table.
                    Defaults to "heParamsTable.txt".
            faces_params_filename (str, optional): Filename for the faces parameters table.
                    Defaults to "faceParamsTable.txt".
            constants_filename (str, optional): Filename for width/height/MAX_EDGES_IN_ANY_FACE.
                    Defaults to "constants.txt".
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)
        np.savetxt(dirpath / vertices_filename, self.vertices)
        np.savetxt(dirpath / edges_filename, self.edges)
        np.savetxt(dirpath / faces_filename, self.faces)
        np.savetxt(dirpath / vertices_params_filename, self.vertices_params)
        np.savetxt(dirpath / edges_params_filename, self.edges_params)
        np.savetxt(dirpath / faces_params_filename, self.faces_params)
        with (dirpath / constants_filename).open("w") as f:
            f.write(f"{self.width} {self.height} {self.MAX_EDGES_IN_ANY_FACE}")

    @classmethod
    def load_mesh_txt(
        cls,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> Self:
        """Load a mesh from text files.

        Only load the vertices, edges and faces, not other parameters.

        Args:
            directory (str): Directory where the text files are stored.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
            vertices_params_filename (str, optional): Filename for the vertices parameters table.
                    Defaults to "vertParamsTable.txt".
            edges_params_filename (str, optional): Filename for the half-edges parameters table.
                    Defaults to "heParamsTable.txt".
            faces_params_filename (str, optional): Filename for the faces parameters table.
                    Defaults to "faceParamsTable.txt".
            constants_filename (str, optional): Filename for width/height/MAX_EDGES_IN_ANY_FACE.
                    Defaults to "constants.txt".

        Returns:
            Self: The loaded mesh.
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        mesh = cls._create()
        # Index tables are WRITTEN by `np.savetxt` in float notation ('0.00...e+00'), so reading
        # them with `dtype=np.int64` makes numpy parse an integer via a float -- a
        # DeprecationWarning that this project's `filterwarnings = ["error"]` turns into a hard
        # error, which is why this loader had no test. Read as float and cast, which also keeps
        # every file written by earlier versions readable.
        mesh.vertices = jnp.array(np.loadtxt(dirpath / vertices_filename, dtype=np.float64))
        mesh.edges = jnp.array(np.loadtxt(dirpath / edges_filename, dtype=np.float64).astype(np.int64))
        mesh.faces = jnp.array(np.loadtxt(dirpath / faces_filename, dtype=np.float64).astype(np.int64))
        # `edges_params` and `faces_params` are PHYSICAL PARAMETERS (tensions, target areas,
        # shape factors), never indices, so they must be read as float: reading them with
        # `dtype=np.int64` silently truncates -- a tension field of mean 1 and spread 0.1
        # round-trips to all-ones, destroying the whole signal. See tests/test_mesh_txt_roundtrip.py.
        mesh.vertices_params = jnp.array(np.loadtxt(dirpath / vertices_params_filename, dtype=np.float64))
        mesh.edges_params = jnp.array(np.loadtxt(dirpath / edges_params_filename, dtype=np.float64))
        mesh.faces_params = jnp.array(np.loadtxt(dirpath / faces_params_filename, dtype=np.float64))
        with (dirpath / constants_filename).open("r") as f:
            numbers = f.readline().split()
            mesh.width = float(numbers[0])
            mesh.height = float(numbers[1])
            mesh.MAX_EDGES_IN_ANY_FACE = int(numbers[2])
        return mesh

    def get_length(self, half_edge_id: Array) -> Array:
        """Get the length of given edges."""

        def _get_length(half_edge_id: Array) -> Array:
            return get_length(half_edge_id, self.vertices, self.edges, self.faces, self.width, self.height)

        return jax.vmap(_get_length)(half_edge_id)

    def get_length_with_offset(self, half_edge_id: Array) -> Array:
        """Get the length of given edges along with its offsets in an array (length, offset x, offset y)."""

        def _get_length_with_offset(half_edge_id: Array) -> Array:
            return get_length_with_offset(half_edge_id, self.vertices, self.edges, self.faces, self.width, self.height)

        return jax.vmap(_get_length_with_offset)(half_edge_id)

    def get_perimeter(self, face_id: Array) -> Array:
        """Get the perimeter of given faces."""

        def _get_perimeter(face_id: Array) -> Array:
            return get_perimeter(
                face_id, self.vertices, self.edges, self.faces, self.width, self.height, self.MAX_EDGES_IN_ANY_FACE
            )

        return jax.vmap(_get_perimeter)(face_id)

    def get_area(self, face_id: Array) -> Array:
        """Get the area of given faces."""

        def _get_area(face_id: Array) -> Array:
            return get_area(
                face_id, self.vertices, self.edges, self.faces, self.width, self.height, self.MAX_EDGES_IN_ANY_FACE
            )

        return jax.vmap(_get_area)(face_id)

    def update_boundary_conditions(self) -> None:
        """Force periodic boundary conditions again after an update.

        Most likely you'll never have to use it yourself, as it can be made automatically with a `PbcBilevelOptimizer`.
        """
        self.vertices, self.edges, self.faces = update_pbc(
            self.vertices, self.edges, self.faces, self.width, self.height
        )


def _make_periodic(  # noqa: C901
    seeds: Array,
    width: float,
    height: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    list[set[int]],
]:
    n_cells = len(seeds)

    # PERIODIC VORONOI - VERTICES EDGES FACES

    if n_cells < 20:
        print("\nWarning: [n_cells < 20] initial condition may not work as expected.\n")

    # add eight neighbor copies of the seeds
    padded_seeds = np.concatenate(
        (
            seeds,
            np.add(seeds, np.full((n_cells, 2), [-width, +height])),
            np.add(seeds, np.full((n_cells, 2), [0, +height])),
            np.add(seeds, np.full((n_cells, 2), [width, +height])),
            np.add(seeds, np.full((n_cells, 2), [-width, 0])),
            np.add(seeds, np.full((n_cells, 2), [width, 0])),
            np.add(seeds, np.full((n_cells, 2), [-width, -height])),
            np.add(seeds, np.full((n_cells, 2), [0, -height])),
            np.add(seeds, np.full((n_cells, 2), [width, -height])),
        ),
        axis=0,
    )

    voronoi = Voronoi(padded_seeds)

    vertices = voronoi.vertices
    edges = voronoi.ridge_vertices
    faces = voronoi.regions

    # original vertices and not from neighbor copies
    col0_mask = (vertices[:, 0] >= 0.0) & (vertices[:, 0] <= width)
    col1_mask = (vertices[:, 1] >= 0.0) & (vertices[:, 1] <= height)

    periodic_voronoi_vertices_idx: NDArray[np.int32] = np.arange(len(vertices))[col0_mask & col1_mask]  # ty:ignore[invalid-assignment]
    periodic_voronoi_vertices_pos: NDArray[np.float64] = vertices[col0_mask & col1_mask]

    edges_inside = []
    edges_outside = []
    offsets_inside = {}
    offsets_outside = {}
    visited = []

    for e in edges:
        source_in = e[0] in periodic_voronoi_vertices_idx
        target_in = e[1] in periodic_voronoi_vertices_idx
        if source_in and target_in:
            edges_inside.append(tuple(sorted((e[0], e[1]))))
            offsets_inside[(e[0], e[1])] = (0, 0)
            offsets_inside[(e[1], e[0])] = (0, 0)
        elif source_in:  # and not target_in
            if vertices[e[1]][0] < 0.0:
                x = vertices[e[1]][0] + width
                offset_x1 = -1
            elif vertices[e[1]][0] > width:
                x = vertices[e[1]][0] - width
                offset_x1 = 1
            else:
                x = vertices[e[1]][0]
                offset_x1 = 0
            if vertices[e[1]][1] < 0.0:
                y = vertices[e[1]][1] + height
                offset_y1 = -1
            elif vertices[e[1]][1] > height:
                y = vertices[e[1]][1] - height
                offset_y1 = 1
            else:
                y = vertices[e[1]][1]
                offset_y1 = 0
            for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                    edges_outside.append(tuple(sorted((e[0], idx))))
                    if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                        offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                        offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                        visited.append((e[0], e[1]))
                        visited.append((e[1], e[0]))
                    break
        elif target_in:  # and not source_in
            if vertices[e[0]][0] < 0.0:
                x = vertices[e[0]][0] + width
                offset_x0 = -1
            elif vertices[e[0]][0] > width:
                x = vertices[e[0]][0] - width
                offset_x0 = 1
            else:
                x = vertices[e[0]][0]
                offset_x0 = 0
            if vertices[e[0]][1] < 0.0:
                y = vertices[e[0]][1] + height
                offset_y0 = -1
            elif vertices[e[0]][1] > height:
                y = vertices[e[0]][1] - height
                offset_y0 = 1
            else:
                y = vertices[e[0]][1]
                offset_y0 = 0
            for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                    edges_outside.append(tuple(sorted((idx, e[1]))))
                    if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                        offsets_outside[(idx, e[1])] = (-offset_x0, -offset_y0)
                        offsets_outside[(e[1], idx)] = (offset_x0, offset_y0)
                        visited.append((e[0], e[1]))
                        visited.append((e[1], e[0]))
                    break

    periodic_voronoi_edges: list[tuple[int, int]] = list(set(edges_inside)) + list(set(edges_outside))

    offsets: dict[tuple[int, int], tuple[int, int]] = offsets_inside | offsets_outside

    faces_inside = []
    faces_inside_outside = []
    for face in faces:
        if face:
            if all(item in periodic_voronoi_vertices_idx for item in face):
                faces_inside.append(tuple(sorted(face)))
            if any(item in face for item in periodic_voronoi_vertices_idx):
                face_inside_outside = []
                for f in face:
                    if f in periodic_voronoi_vertices_idx:
                        face_inside_outside.append(f)
                    else:
                        if vertices[f][0] < 0.0:
                            x = vertices[f][0] + width
                        elif vertices[f][0] > width:
                            x = vertices[f][0] - width
                        else:
                            x = vertices[f][0]
                        if vertices[f][1] < 0.0:
                            y = vertices[f][1] + height
                        elif vertices[f][1] > height:
                            y = vertices[f][1] - height
                        else:
                            y = vertices[f][1]
                        for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                            if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                                face_inside_outside.append(idx)
                                break

                faces_inside_outside.append(tuple(sorted(face_inside_outside)))

    periodic_voronoi_faces: list[set[int]] = list(set(faces_inside_outside))
    return (
        periodic_voronoi_vertices_idx,
        periodic_voronoi_vertices_pos,
        periodic_voronoi_edges,
        offsets,
        periodic_voronoi_faces,
    )


class _FaceOrderingError(RuntimeError):
    """The per-face edge-ordering walk in `_make_he_structure` cannot terminate.

    Subclasses `RuntimeError` so that any caller already catching `RuntimeError` still catches this.
    Private: it is a diagnostic for a construction that can hang, not a new public API.
    """


def _make_he_structure(  # noqa: C901
    width: float,
    height: float,
    periodic_vertices_idx: NDArray[np.int32],
    periodic_vertices_positions: NDArray[np.float64],
    periodic_edges: list[tuple[int, int]],
    offsets: dict[tuple[int, int], tuple[int, int]],
    periodic_faces: list[set[int]],
    vertices_offset: tuple[float, float] = (0, 0),
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return half-edge structure with vertices, edges, faces.

    vertices is the position of vertices.
    edges is a table of int with:
        previous half-edge id in face,
        next half-edge id in face,
        twin half-edge id,
        source vertex id,
        target vertex id,
        face it belongs id,
        offset x,
        offset y.
    faces records just for each face the id of one of its edges.
    """
    # HALF EDGE DATA STRUCTURE

    # Reciprocating edges
    periodic_half_edges = []
    for e in periodic_edges:
        periodic_half_edges.append(e)
        periodic_half_edges.append((e[1], e[0]))

    # Finding clockwise (or counterclockwise) half edge set for each face
    ordered_edges_periodic_faces = []
    for face_index, face in enumerate(periodic_faces):
        edges_face = [(f1, f2) for f1 in face for f2 in face if (f1, f2) in periodic_edges]
        i = 0
        start_edge = edges_face[i]
        ordered_face = [start_edge]
        e = start_edge
        visited = [e]
        # BOUND AND RAISE. The walk advances with `edges_face[i % len(face)]`: the
        # modulus is taken over `face` -- the face's VERTICES -- while the list being indexed is
        # `edges_face`, the vertex pairs of that face that are also edges. Termination requires
        # every element of `edges_face` to be visited, but only indices 0 .. len(face)-1 are
        # reachable, so when `len(edges_face) > len(face)` the condition can never be satisfied and
        # this loop runs forever -- with no exception, no warning, and no output.
        #
        # The bound below cannot change any mesh that currently builds. Termination already implies
        # `len(edges_face) == len(face)` (each of the n vertices' induced edges must be reachable
        # within n indices, and an n-gon's vertex set induces at least its own n boundary edges), and
        # a terminating walk needs at most O(len(face)^2) steps because each of the n appends costs
        # at most one cyclic scan. Measured, not just deduced: the invariant
        # `len(edges_face) == len(face)` holds on every seed case and every face of every committed
        # synthetic mesh that builds, and fails on every case that hangs; see
        # tests/test_pbc_mesh_face_ordering.py.
        #
        # This converts an unbounded silent hang into an actionable error. It deliberately does NOT
        # change the modulus: where a chord exists, walking `edges_face` can emit a truncated or
        # self-intersecting `ordered_face`, and `geo.get_area` takes `0.5*|shoelace|`, which accepts
        # both silently -- the `get_area_bounded` failure mode. Fixing that needs a geometric
        # decision, with an acceptance test on `clamped.signed_face_areas`, not `|shoelace|`.
        max_steps = 8 * len(face) ** 2 + 64
        steps = 0
        while sorted(edges_face) != sorted(visited):
            if steps >= max_steps:
                surplus = len(edges_face) - len(face)
                msg = (
                    f"PbcMesh face ordering did not terminate for face index {face_index} after "
                    f"{max_steps} steps. len(face) = {len(face)} vertices but len(edges_face) = "
                    f"{len(edges_face)} induced edges; termination requires these to be EQUAL, "
                    f"because the walk advances with `edges_face[i % len(face)]` and can only reach "
                    f"indices 0..{len(face) - 1}. "
                    + (f"The {surplus} surplus entries are CHORDS: edges joining two vertices of "
                       f"this face that are not its own boundary edges -- what a degenerate Voronoi "
                       f"cell produces at low cell counts. " if surplus > 0 else
                       f"The deficit is {-surplus}: a boundary edge of this face is missing from "
                       f"the edge list. ")
                    + f"Offending face vertex set: {sorted(face)}. Box {width} x {height}. "
                    f"Measured incidence: 0/189 seed cases that build "
                    f"violate the invariant, 211/211 that hang do, and nb_seeds >= 50 did not hang "
                    f"in 120 cases -- so raise nb_seeds, move the aspect ratio toward 1:1, or pick "
                    f"another random_key."
                )
                raise _FaceOrderingError(msg)
            steps += 1
            if e[0] == start_edge[1] and e not in visited:
                ordered_face.append(e)
                start_edge = e
                visited.append(e)
            if e[1] == start_edge[1] and e not in visited:
                ordered_face.append((e[1], e[0]))
                start_edge = (e[1], e[0])
                visited.append(e)
            i += 1
            e = edges_face[i % len(face)]

        order = 0
        sum0_offsets = 0
        sum1_offsets = 0
        points = []
        for e in ordered_face:
            idx0 = list(periodic_vertices_idx).index(e[0])
            idx1 = list(periodic_vertices_idx).index(e[1])
            e_offsets = offsets[e]

            prev_sum0_offsets = sum0_offsets
            prev_sum1_offsets = sum1_offsets
            sum0_offsets += e_offsets[0]
            sum1_offsets += e_offsets[1]

            order += (
                (periodic_vertices_positions[idx1][0] + sum0_offsets * width)
                - (periodic_vertices_positions[idx0][0] + prev_sum0_offsets * width)
            ) * (
                (periodic_vertices_positions[idx1][1] + sum1_offsets * height)
                + (periodic_vertices_positions[idx0][1] + prev_sum1_offsets * height)
            )

            points.append(
                (
                    periodic_vertices_positions[idx0][0] + prev_sum0_offsets * width,
                    periodic_vertices_positions[idx0][1] + prev_sum1_offsets * height,
                )
            )

            points.append(
                (
                    periodic_vertices_positions[idx1][0] + sum0_offsets * width,
                    periodic_vertices_positions[idx1][1] + sum1_offsets * height,
                )
            )

        if order < 0:
            ordered_edges_periodic_faces.append(ordered_face)
        if order > 0:
            new_ordered_face = [(e[1], e[0]) for e in reversed(ordered_face)]
            ordered_edges_periodic_faces.append(new_ordered_face)
        if order == 0:
            print("\nError: no order detected for face " + str(face) + "\n")
            exit()

    # VERT FACE HE TABLES

    vertTable = periodic_vertices_positions - vertices_offset

    faceTable = np.zeros(len(periodic_faces), dtype=np.int32)
    for i, hedges_face in enumerate(ordered_edges_periodic_faces):
        for j, he in enumerate(periodic_half_edges):
            if he == hedges_face[0]:
                faceTable[i] = j  # he_inside

    heTable = np.zeros((len(periodic_half_edges), 8), dtype=np.int32)
    for i, he in enumerate(periodic_half_edges):
        for hedges_face in ordered_edges_periodic_faces:
            if he in hedges_face:
                idx = hedges_face.index(he)
                heTable[i][0] = periodic_half_edges.index(hedges_face[(idx - 1) % len(hedges_face)])  # he_prev
                heTable[i][1] = periodic_half_edges.index(hedges_face[(idx + 1) % len(hedges_face)])  # he_next
                heTable[i][3] = list(periodic_vertices_idx).index(he[0])  # vert source
                heTable[i][4] = list(periodic_vertices_idx).index(he[1])  # vert target
                heTable[i][5] = ordered_edges_periodic_faces.index(hedges_face)  # face
                break
        heTable[i][2] = periodic_half_edges.index((he[1], he[0]))  # he twin
        heTable[i][6] = offsets[he][0]  # he_offset x vert target
        heTable[i][7] = offsets[he][1]  # he_offset y vert target
    return (vertTable, heTable, faceTable)
