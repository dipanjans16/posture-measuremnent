import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import plotly
import plotly.graph_objects as go
import plotly.express as px
import trimesh
import argparse
import smplx
import json
import torch

# import ipywidgets as widgets
from plotly.subplots import make_subplots


from measure.measurement_definitions import MeasurementType
from measure.measure_utils import convex_hull_from_3D_points, filter_body_part_slices
from measure.joint_definitions import SMPL_IND2JOINT, SMPLX_IND2JOINT
from measure.landmark_definitions import SMPL_LANDMARK_INDICES, SMPLX_LANDMARK_INDICES


class Visualizer:
    """
    Creates interactive 3D visualizations of body models with measurements, landmarks and joints.
    All measurements are in centimeters.

    Args:
        verts (np.ndarray): Vertex coordinates of shape (N,3)
        faces (np.ndarray): Face indices of shape (F,3) defining mesh triangles
        joints (np.ndarray): Joint coordinates of shape (J,3)
        landmarks (dict): Maps landmark names to vertex indices
        measurements (dict): Maps measurement names to their values in cm
        measurement_types (dict): Maps measurement names to their types (length/circumference)
        length_definitions (dict): Maps length measurement names to pairs of landmark indices
        circumf_definitions (dict): Maps circumference names to required landmarks and joints
        joint2ind (dict): Maps joint names to their indices
        circumf_2_bodypart (dict): Maps circumference names to body part names
        face_segmentation (dict): Maps body part names to face indices
        visualize_body (bool, optional): Whether to show the body mesh. Defaults to True.
        visualize_landmarks (bool, optional): Whether to show landmarks. Defaults to True.
        visualize_joints (bool, optional): Whether to show joints. Defaults to True.
        visualize_measurements (bool, optional): Whether to show measurements. Defaults to True.
        title (str, optional): Plot title. Defaults to "Measurement visualization".
    """

    def __init__(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray,
        landmarks: dict,
        measurements: dict,
        measurement_types: dict,
        length_definitions: dict,
        circumf_definitions: dict,
        joint2ind: dict,
        circumf_2_bodypart: dict,
        face_segmentation: dict,
        visualize_body: bool = True,
        visualize_landmarks: bool = True,
        visualize_joints: bool = True,
        visualize_measurements: bool = True,
        title: str = "Measurement visualization",
    ):

        self.verts = verts
        self.faces = faces
        self.joints = joints
        self.landmarks = landmarks
        self.measurements = measurements
        self.measurement_types = measurement_types
        self.length_definitions = length_definitions
        self.circumf_definitions = circumf_definitions
        self.joint2ind = joint2ind
        self.circumf_2_bodypart = circumf_2_bodypart
        self.face_segmentation = face_segmentation

        self.visualize_body = visualize_body
        self.visualize_landmarks = visualize_landmarks
        self.visualize_joints = visualize_joints
        self.visualize_measurements = visualize_measurements

        self.title = title

    @staticmethod
    def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
        """
        Creates a 3D mesh visualization of the body model.

        Args:
            verts (np.ndarray): Vertex coordinates of shape (N,3)
            faces (np.ndarray): Face indices of shape (F,3) defining mesh triangles

        Returns:
            plotly.graph_objs.Mesh3d: 3D mesh plot object
        """
        mesh_plot = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            color="gray",
            hovertemplate="<i>Index</i>: %{text}",
            text=[i for i in range(verts.shape[0])],
            # i, j and k give the vertices of triangles
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.6,
            name="body",
        )
        return mesh_plot

    @staticmethod
    def create_joint_plot(joints: np.ndarray) -> plotly.graph_objs.Scatter3d:

        return go.Scatter3d(
            x=joints[:, 0],
            y=joints[:, 1],
            z=joints[:, 2],
            mode="markers",
            marker=dict(size=8, color="black", opacity=1, symbol="cross"),
            name="joints",
        )

    @staticmethod
    def create_wireframe_plot(
        verts: np.ndarray, faces: np.ndarray
    ) -> plotly.graph_objs.Scatter3d:
        """
        Creates a wireframe visualization of the mesh using line segments.

        Args:
            verts (np.ndarray): Vertex coordinates of shape (N,3)
            faces (np.ndarray): Face indices of shape (F,3) defining mesh triangles

        Returns:
            plotly.graph_objs.Scatter3d: 3D wireframe plot object
        """
        i = faces[:, 0]
        j = faces[:, 1]
        k = faces[:, 2]

        triangles = np.vstack((i, j, k)).T

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        vertices = np.vstack((x, y, z)).T
        tri_points = vertices[triangles]

        # extract the lists of x, y, z coordinates of the triangle
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])

        # return Xe, Ye, Ze
        wireframe = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            name="wireframe",
            line=dict(color="rgb(70,70,70)", width=1),
        )
        return wireframe

    def create_landmarks_plot(
        self, landmark_names: List[str], verts: np.ndarray
    ) -> List[plotly.graph_objs.Scatter3d]:
        """
        Creates 3D scatter plots for specified landmarks on the body model.

        Args:
            landmark_names (List[str]): Names of landmarks to visualize
            verts (np.ndarray): Vertex coordinates of shape (N,3)

        Returns:
            List[plotly.graph_objs.Scatter3d]: List of 3D scatter plots for landmarks
        """

        plots = []

        landmark_colors = dict(
            zip(self.landmarks.keys(), px.colors.qualitative.Alphabet)
        )

        for lm_name in landmark_names:
            if lm_name not in self.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            lm_index = self.landmarks[lm_name]
            if isinstance(lm_index, tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index]

            plot = go.Scatter3d(
                x=[lm[0]],
                y=[lm[1]],
                z=[lm[2]],
                mode="markers",
                marker=dict(
                    size=8,
                    color=landmark_colors[lm_name],
                    opacity=1,
                ),
                name=lm_name,
            )

            plots.append(plot)

        return plots

    def create_measurement_length_plot(
        self, measurement_name: str, verts: np.ndarray, color: str
    ) -> plotly.graph_objs.Scatter3d:
        """
        Creates a 3D line plot for length-based measurements.

        Args:
            measurement_name (str): Name of the measurement to visualize
            verts (np.ndarray): Vertex coordinates of shape (N,3)
            color (str): Color for the measurement line

        Returns:
            plotly.graph_objs.Scatter3d: 3D line plot for the measurement
        """

        if measurement_name not in self.length_definitions.keys():
            print(f"Length measurement {measurement_name} is not defined.")
            return

        lm1, lm2 = self.length_definitions[measurement_name]

        if lm1 not in self.landmarks.keys():
            print(f"Landmark {lm1} not defined.")
            return
        if lm2 not in self.landmarks.keys():
            print(f"Landmark {lm2} not defined.")
            return

        lm1_ind = self.landmarks[lm1]
        lm2_ind = self.landmarks[lm2]

        lm1_verts = verts[lm1_ind, :]
        lm2_verts = verts[lm2_ind, :]

        x_coords = [lm1_verts[0], lm2_verts[0]]
        y_coords = [lm1_verts[1], lm2_verts[1]]
        z_coords = [lm1_verts[2], lm2_verts[2]]

        measurement_line = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines+markers",
            line=dict(color=color, width=10),
            marker=dict(
                size=10,
                color=color,
            ),
            name=measurement_name,
        )
        return measurement_line

    def create_measurement_circumference_plot(
        self, measurement_name: str, verts: np.ndarray, faces: np.ndarray, color: str
    ) -> Optional[plotly.graph_objs.Scatter3d]:
        """
        Creates a 3D line plot for circumference-based measurements.

        Args:
            measurement_name (str): Name of the circumference measurement
            verts (np.ndarray): Vertex coordinates of shape (N,3)
            faces (np.ndarray): Face indices defining mesh triangles
            color (str): Color for the circumference line

        Returns:
            Optional[plotly.graph_objs.Scatter3d]: 3D line plot for the circumference
        """

        if measurement_name not in self.circumf_definitions.keys():
            print(f"Circumference measurement {measurement_name} is not defined.")
            return

        circumf_def = self.circumf_definitions[measurement_name]

        if "joints" in circumf_def:
            jnt_names = circumf_def["joints"]
        else:
            jnt_names = circumf_def["joint"]

        landmarks = circumf_def["landmarks"]

        plane_joints = []
        for jnt in jnt_names:
            plane_joints.append(self.joints[self.joint2ind[jnt], :])
        plane_joints = np.array(plane_joints)

        if (
            self.circumf_2_bodypart[measurement_name]
            not in self.face_segmentation.keys()
        ):
            print(
                f"Body part {self.circumf_2_bodypart[measurement_name]} not in face segmentation."
            )
            return

        # Get body part slice
        slice_faces = self.face_segmentation[self.circumf_2_bodypart[measurement_name]]
        slice_verts = np.unique(faces[slice_faces])
        slice_verts_coordinates = verts[slice_verts, :]

        # Filter to relevant parts
        slice_segments = filter_body_part_slices(
            slice_verts_coordinates,
            landmarks,
            verts,
            self.landmarks,
            plane_joints,
            measurement_name,
        )
        if len(slice_segments) == 0:
            print(f"No slice segments found for {measurement_name}")
            return

        # Get convex hull
        circumf_verts = convex_hull_from_3D_points(slice_segments)
        circumf_verts = np.append(
            circumf_verts, [circumf_verts[0]], axis=0
        )  # close the circumference

        circumference_line = go.Scatter3d(
            x=circumf_verts[:, 0],
            y=circumf_verts[:, 1],
            z=circumf_verts[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=10),
            marker=dict(
                size=5,
                color=color,
            ),
            name=measurement_name,
        )
        return circumference_line

    def visualize(
        self,
        measurement_names: List[str] = [],
        landmark_names: List[str] = [],
        title: str = "Measurement visualization",
    ) -> plotly.graph_objects.Figure:
        """
        Creates a comprehensive 3D visualization of the body model.

        Args:
            measurement_names (List[str], optional): Measurements to visualize. Defaults to [].
            landmark_names (List[str], optional): Landmarks to visualize. Defaults to [].
            title (str, optional): Plot title. Defaults to "Measurement visualization".

        Returns:
            plotly.graph_objects.Figure: Interactive 3D figure
        """

        if not measurement_names:
            measurement_names = list(self.measurements.keys())

        if not landmark_names:
            landmark_names = list(self.landmarks.keys())

        fig = go.Figure()

        if self.visualize_body:
            mesh_plot = Visualizer.create_mesh_plot(self.verts, self.faces)
            fig.add_trace(mesh_plot)

        # wireframe = Visualizer.create_wireframe_plot(verts,faces)
        # fig.add_trace(wireframe)

        if self.visualize_joints:
            joint_plot = Visualizer.create_joint_plot(self.joints)
            fig.add_trace(joint_plot)

        if self.visualize_landmarks:
            landmark_plots = self.create_landmarks_plot(landmark_names, self.verts)
            for lm_plot in landmark_plots:
                fig.add_trace(lm_plot)

        if self.visualize_measurements:
            measurement_colors = dict(
                zip(measurement_names, px.colors.qualitative.Dark24)
            )

            for m_name in measurement_names:

                if m_name not in self.measurements.keys():
                    print(f"Measurement {m_name} not computed.")
                    continue

                if self.measurement_types[m_name] == MeasurementType().LENGTH:
                    measurement_plot = self.create_measurement_length_plot(
                        m_name, self.verts, measurement_colors[m_name]
                    )
                    fig.add_trace(measurement_plot)

                elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                    measurement_plot = self.create_measurement_circumference_plot(
                        m_name, self.verts, self.faces, measurement_colors[m_name]
                    )
                    if measurement_plot is not None:
                        fig.add_trace(measurement_plot)

        fig.update_layout(
            scene_aspectmode="data",
            width=1000,
            height=700,
            title=title,
        )

        return fig


def viz_smplx_joints(
    visualize_body: bool = True,
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPLX joints",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model = smplx.create(
        model_path="data",
        model_type="smplx",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        # body_pose=torch.zeros((1, (55-1) * 3)),
        ext="pkl",
    )

    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_joints = smplx_model.joints.detach().numpy()[0]
    smplx_joint_pelvis = smplx_joints[0, :]
    smplx_joints = smplx_joints - smplx_joint_pelvis
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]
    smplx_vertices = smplx_vertices - smplx_joint_pelvis
    smplx_faces = smplx.SMPLX("data/smplx", ext="pkl").faces

    joint_colors = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + ["#000000"]
    )

    if isinstance(fig, type(None)):
        fig = go.Figure()

    for i in range(smplx_joints.shape[0]):

        if i in SMPLX_IND2JOINT.keys():
            joint_name = SMPLX_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(
            x=[smplx_joints[i, 0]],
            y=[smplx_joints[i, 1]],
            z=[smplx_joints[i, 2]],
            mode="markers",
            marker=dict(size=10, color=joint_colors[i], opacity=1, symbol="circle"),
            name="smplx-" + joint_name,
        )

        fig.add_trace(joint_plot)

    if visualize_body:
        plot_body = go.Mesh3d(
            x=smplx_vertices[:, 0],
            y=smplx_vertices[:, 1],
            z=smplx_vertices[:, 2],
            color="red",
            i=smplx_faces[:, 0],
            j=smplx_faces[:, 1],
            k=smplx_faces[:, 2],
            name="smplx mesh",
            showscale=True,
            opacity=0.5,
        )
        fig.add_trace(plot_body)

    fig.update_layout(
        scene_aspectmode="data",
        width=1000,
        height=700,
        title=title,
    )

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_joints(
    visualize_body: bool = True,
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPL joints",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_joints = smpl_model.joints.detach().numpy()[0]
    smpl_joints_pelvis = smpl_joints[0, :]
    smpl_joints = smpl_joints - smpl_joints_pelvis
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]
    smpl_vertices = smpl_vertices - smpl_joints_pelvis
    smpl_faces = smplx.SMPL("data/smpl", ext="pkl").faces

    joint_colors = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + ["#000000"]
    )

    if isinstance(fig, type(None)):
        fig = go.Figure()

    for i in range(smpl_joints.shape[0]):

        if i in SMPL_IND2JOINT.keys():
            joint_name = SMPL_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(
            x=[smpl_joints[i, 0]],
            y=[smpl_joints[i, 1]],
            z=[smpl_joints[i, 2]],
            mode="markers",
            marker=dict(size=10, color=joint_colors[i], opacity=1, symbol="cross"),
            name="smpl-" + joint_name,
        )

        fig.add_trace(joint_plot)

    if visualize_body:
        plot_body = go.Mesh3d(
            x=smpl_vertices[:, 0],
            y=smpl_vertices[:, 1],
            z=smpl_vertices[:, 2],
            # facecolor=face_colors,
            color="blue",
            i=smpl_faces[:, 0],
            j=smpl_faces[:, 1],
            k=smpl_faces[:, 2],
            name="smpl mesh",
            showscale=True,
            opacity=0.5,
        )
        fig.add_trace(plot_body)

    fig.update_layout(
        scene_aspectmode="data",
        width=1000,
        height=700,
        title=title,
    )
    if show:
        fig.show()
    else:
        return fig


def viz_face_segmentation(
    verts: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    title: str = "Segmented body",
    name: str = "mesh",
    show: bool = True,
) -> Optional[plotly.graph_objects.Figure]:
    """
    Creates a 3D visualization of the body mesh with faces colored by segmentation.

    Args:
        verts (np.ndarray): Vertex coordinates of shape (N,3)
        faces (np.ndarray): Face indices of shape (F,3) defining mesh triangles
        face_colors (np.ndarray): RGB colors for each face of shape (F,3)
        title (str, optional): Plot title. Defaults to "Segmented body".
        name (str, optional): Name for the mesh trace. Defaults to "mesh".
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        plotly.graph_objs.Figure: Figure object if show=False, None otherwise
    """

    fig = go.Figure()
    mesh_plot = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        facecolor=face_colors,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        showscale=True,
        opacity=1,
    )
    fig.add_trace(mesh_plot)

    fig.update_layout(
        scene_aspectmode="data",
        width=1000,
        height=700,
        title=title,
    )

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_face_segmentation(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPL face segmentation",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPL model with face segmentation coloring.

    Args:
        fig: Existing plotly figure to add to, creates new if None
        show: Whether to display the figure immediately
        title: Title for the visualization

    Returns:
        Figure object if show=False, None otherwise
    """

    with open("measure/smpl_face_segmentation.json") as f:
        face_segmentation = json.load(f)

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]
    smpl_faces = smplx.SMPL("data/smpl", ext="pkl").faces

    # Create face colors
    part_colors = px.colors.qualitative.Alphabet
    face_colors = np.zeros((smpl_faces.shape[0], 3))
    for i, (part_name, face_indices) in enumerate(face_segmentation.items()):
        color = px.colors.hex_to_rgb(part_colors[i % len(part_colors)])
        face_colors[face_indices] = color

    viz_face_segmentation(
        smpl_vertices, smpl_faces, face_colors, title=title, name="smpl mesh", show=show
    )


def viz_smplx_face_segmentation(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPLX face segmentation",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPLX model with face segmentation coloring.

    Args:
        fig: Existing plotly figure to add to, creates new if None
        show: Whether to display the figure immediately
        title: Title for the visualization

    Returns:
        Figure object if show=False, None otherwise
    """

    with open("measure/smplx_face_segmentation.json") as f:
        face_segmentation = json.load(f)

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model = smplx.create(
        model_path="data",
        model_type="smplx",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]
    smplx_faces = smplx.SMPLX("data/smplx", ext="pkl").faces

    # Create face colors
    part_colors = px.colors.qualitative.Alphabet
    face_colors = np.zeros((smplx_faces.shape[0], 3))
    for i, (part_name, face_indices) in enumerate(face_segmentation.items()):
        color = px.colors.hex_to_rgb(part_colors[i % len(part_colors)])
        face_colors[face_indices] = color

    viz_face_segmentation(
        smplx_vertices,
        smplx_faces,
        face_colors,
        title=title,
        name="smplx mesh",
        show=show,
    )


def viz_point_segmentation(
    verts: np.ndarray,
    point_segm: Dict[str, List[int]],
    title: str = "Segmented body",
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes point segmentation on a 3D body model.

    Args:
        verts: Vertex coordinates of shape (N,3)
        point_segm: Dictionary mapping body part names to vertex indices
        title: Plot title
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately

    Returns:
        Figure object if show=False, None otherwise
    """

    if isinstance(fig, type(None)):
        fig = go.Figure()

    part_colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24

    for i, (part_name, point_indices) in enumerate(point_segm.items()):

        points_plot = go.Scatter3d(
            x=verts[point_indices, 0],
            y=verts[point_indices, 1],
            z=verts[point_indices, 2],
            mode="markers",
            marker=dict(size=5, color=part_colors[i], opacity=1, symbol="circle"),
            name=part_name,
        )

        fig.add_trace(points_plot)

    fig.update_layout(
        scene_aspectmode="data",
        width=1000,
        height=700,
        title=title,
    )

    if show:
        fig.show()
    else:
        return fig


def viz_smplx_point_segmentation(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPLX point segmentation",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPLX model with point segmentation.

    Args:
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately
        title: Plot title

    Returns:
        Figure object if show=False, None otherwise
    """
    from measure.smplx_point_segmentation import SMPLX_POINT_SEGM

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model = smplx.create(
        model_path="data",
        model_type="smplx",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]

    viz_point_segmentation(
        smplx_vertices, SMPLX_POINT_SEGM, title=title, fig=fig, show=show
    )


def viz_smpl_point_segmentation(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPL point segmentation",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPL model with point segmentation.

    Args:
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately
        title: Plot title

    Returns:
        Figure object if show=False, None otherwise
    """
    from measure.smpl_point_segmentation import SMPL_POINT_SEGM

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]

    viz_point_segmentation(
        smpl_vertices, SMPL_POINT_SEGM, title=title, fig=fig, show=show
    )


def viz_landmarks(
    verts: np.ndarray,
    landmark_dict: Dict[str, int],
    title: str = "Visualize landmarks",
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    name: str = "points",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes landmarks on a 3D body model.

    Args:
        verts: Vertex coordinates of shape (N,3)
        landmark_dict: Dictionary mapping landmark names to vertex indices
        title: Plot title
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately
        name: Base name for landmark traces

    Returns:
        Figure object if show=False, None otherwise
    """

    if isinstance(fig, type(None)):
        fig = go.Figure()

    landmark_colors = dict(zip(landmark_dict.keys(), px.colors.qualitative.Alphabet))

    for landmark_name, point_ind in landmark_dict.items():

        if isinstance(point_ind, list):
            point_ind = point_ind[0]

        landmark_plot = go.Scatter3d(
            x=[verts[point_ind, 0]],
            y=[verts[point_ind, 1]],
            z=[verts[point_ind, 2]],
            mode="markers+text",
            marker=dict(
                size=15,
                color=landmark_colors[landmark_name],
                opacity=1,
                symbol="circle",
            ),
            name=f"{name}-{landmark_name}",
            text=[landmark_name],
            textposition="middle center",
        )

        fig.add_trace(landmark_plot)

    fig.update_layout(
        scene_aspectmode="data",
        width=1000,
        height=700,
        title=title,
    )

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_landmarks(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPL landmarks",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPL model landmarks.

    Args:
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately
        title: Plot title

    Returns:
        Figure object if show=False, None otherwise
    """
    betas = torch.zeros((1, 10), dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]

    viz_landmarks(
        smpl_vertices,
        SMPL_LANDMARK_INDICES,
        title=title,
        fig=fig,
        show=show,
        name="smpl",
    )


def viz_smplx_landmarks(
    fig: Optional[plotly.graph_objects.Figure] = None,
    show: bool = True,
    title: str = "SMPLX landmarks",
) -> Optional[plotly.graph_objects.Figure]:
    """
    Visualizes SMPLX model landmarks.

    Args:
        fig: Existing figure to add to, creates new if None
        show: Whether to display immediately
        title: Plot title

    Returns:
        Figure object if show=False, None otherwise
    """
    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model = smplx.create(
        model_path="data",
        model_type="smplx",
        gender="NEUTRAL",
        use_face_contour=False,
        num_betas=10,
        ext="pkl",
    )

    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]

    viz_landmarks(
        smplx_vertices,
        SMPLX_LANDMARK_INDICES,
        title=title,
        fig=fig,
        show=show,
        name="smplx",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize body models, joints and segmentations.."
    )
    parser.add_argument(
        "--visualize_smpl_and_smplx_face_segmentation",
        action="store_true",
        help="Visualize face segmentations for smplx model.",
    )
    parser.add_argument(
        "--visualize_smpl_and_smplx_joints",
        action="store_true",
        help="visualize smpl and smplx joints on same plot.",
    )
    parser.add_argument(
        "--visualize_smpl_and_smplx_point_segmentation",
        action="store_true",
        help="visualize smpl and smplx point segmentation on two separate plots.",
    )
    parser.add_argument(
        "--visualize_smpl_and_smplx_landmarks",
        action="store_true",
        help="visualize smpl and smplx landmarks on two separate plots.",
    )
    args = parser.parse_args()

    if args.visualize_smpl_and_smplx_face_segmentation:
        # mesh is not compatible with subplots so these are plotted
        # onto separate plots
        viz_smpl_face_segmentation(fig=None, show=True)
        viz_smplx_face_segmentation(fig=None, show=True)

    if args.visualize_smpl_and_smplx_joints:
        title = "SMPL and SMPLX joints"
        fig = viz_smpl_joints(visualize_body=True, fig=None, show=False, title=title)
        viz_smplx_joints(visualize_body=True, fig=fig, show=True, title=title)

    if args.visualize_smpl_and_smplx_point_segmentation:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("SMPL", "SMPLX"),
        )
        title = "SMPL and SMPLX point segmentation"

        fig_smpl = viz_smpl_point_segmentation(fig=None, show=False, title=title)
        fig_smplx = viz_smplx_point_segmentation(fig=None, show=False, title=title)

        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i], row=1, col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i], row=1, col=2)

        fig.update_layout(fig_smpl.layout)
        fig.update_layout(
            scene2_aspectmode="data", showlegend=False, width=1200, height=700
        )
        fig.show()

    if args.visualize_smpl_and_smplx_landmarks:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("SMPL", "SMPLX"),
        )
        title = "SMPL and SMPLX landmarks"

        fig_smpl = viz_smpl_landmarks(fig=None, show=False, title=title)
        fig_smplx = viz_smplx_landmarks(fig=None, show=False, title=title)

        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i], row=1, col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i], row=1, col=2)

        fig.update_layout(fig_smpl.layout)
        fig.update_layout(
            scene2_aspectmode="data", showlegend=False, width=1200, height=700
        )
        fig.show()
