#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


MODEL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)


def _build_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        in vec3 color;
        
        out vec3 res_color;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            res_color = color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        in vec3 res_color;
        
        out vec3 out_color;
        void main() {
            out_color = res_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _build_objects_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;
        
        in vec3 position;

        void main() {
            gl_Position = mvp * vec4(position, 1.0);
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        uniform vec3 color;

        out vec3 out_color;
        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """
        self._program = _build_program()
        self._objects_program = _build_objects_program()
        self._tracked_cam_parameters = tracked_cam_parameters
        self._tracked_cam_track = tracked_cam_track
        self._cam_points = np.array([point.t_vec for point in tracked_cam_track], dtype=np.float32)

        self._points_size = len(point_cloud.ids)
        self._points = vbo.VBO(point_cloud.points.reshape(-1).astype(np.float32))
        self._colors = vbo.VBO(point_cloud.colors.reshape(-1).astype(np.float32))

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        view = self._matric_view(-camera_tr_vec, np.linalg.inv(camera_rot_mat))
        project = self._matrix_project(camera_fov_y)
        mvp = project.dot(view.dot(MODEL))

        self._render(mvp)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._render_objects(mvp, GL.GL_LINE_STRIP,
                             vbo.VBO(self._cam_points[:tracked_cam_track_pos + 1].astype(np.float32).reshape(-1)),
                             color, tracked_cam_track_pos + 1)
        frustrum = self._frustrum_points(self._tracked_cam_track[tracked_cam_track_pos],
                                         self._tracked_cam_parameters.fov_y,
                                         self._tracked_cam_parameters.aspect_ratio)
        borders = np.array([[self._cam_points[tracked_cam_track_pos], frust] for frust in frustrum],
                           dtype=np.float32).flatten()
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        self._render_objects(mvp, GL.GL_LINE_LOOP,
                             vbo.VBO(frustrum.astype(np.float32).reshape(-1)),
                             color, 4)
        self._render_objects(mvp, GL.GL_LINES,
                             vbo.VBO(borders.astype(np.float32).reshape(-1)), color, 8)

        GLUT.glutSwapBuffers()

    def _render(self, mvp):
        shaders.glUseProgram(self._program)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._program, 'mvp'),
            1, True, mvp)

        self._colors.bind()
        color_loc = GL.glGetAttribLocation(self._program, 'color')
        GL.glEnableVertexAttribArray(color_loc)
        GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._colors)
        self._points.bind()
        position_loc = GL.glGetAttribLocation(self._objects_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._points)

        GL.glDrawArrays(GL.GL_POINTS, 0, self._points_size)

        GL.glDisableVertexAttribArray(position_loc)
        self._points.unbind()
        GL.glDisableVertexAttribArray(color_loc)
        self._colors.unbind()
        shaders.glUseProgram(0)

    def _render_objects(self, mvp, mode, points, color, num):
        shaders.glUseProgram(self._objects_program)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._objects_program, 'mvp'),
            1, True, mvp)
        GL.glUniform3fv(
            GL.glGetUniformLocation(self._objects_program, 'color'),
            1, color
        )
        points.bind()
        position_loc = GL.glGetAttribLocation(self._objects_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 points)

        GL.glDrawArrays(mode, 0, num)

        GL.glDisableVertexAttribArray(position_loc)
        points.unbind()
        shaders.glUseProgram(0)

    def _matric_view(self, translation, rotation):
        m1 = np.eye(4, dtype=np.float32)
        m2= np.eye(4, dtype=np.float32)
        m1[:3, 3] = translation
        m2[:3, :3] = rotation
        return np.dot(m2, m1)

    def _matrix_project(self, fovy, znear=0.5, zfar=100.):
        aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        ymax = znear * np.tan(fovy / 2.)
        xmax = ymax * aspect_ratio
        delta = znear - zfar

        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = znear / xmax
        m[1, 1] = znear / ymax
        m[2, 2] = (zfar + znear) / delta
        m[2, 3] = 2. * znear * zfar / delta
        m[3, 2] = -1.

        return m

    def _frustrum_points(self, cam_track, fovy, aspect_ratio, zfar=10.):
        ymax = zfar * np.tan(fovy)
        xmax = ymax * aspect_ratio

        m = np.zeros((4, 3), dtype=np.float32)
        m[0] = cam_track.r_mat.dot(np.array([xmax, ymax, zfar])) + cam_track.t_vec
        m[1] = cam_track.r_mat.dot(np.array([xmax, -ymax, zfar])) + cam_track.t_vec
        m[2] = cam_track.r_mat.dot(np.array([-xmax, -ymax, zfar])) + cam_track.t_vec
        m[3] = cam_track.r_mat.dot(np.array([-xmax, ymax, zfar])) + cam_track.t_vec

        return m
