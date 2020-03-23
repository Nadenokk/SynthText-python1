from __future__ import division
import numpy as np 
from ransac import fit_plane_ransac
from sys import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mayavi.mlab as mym


class LUT_RGB(object):
    """
    RGB LUT for Mayavi glyphs.

    RGB LUT для символов майави.
    """
    def __create_8bit_rgb_lut__():
        xl = np.mgrid[0:256, 0:256, 0:256]
        lut = np.vstack((xl[0].reshape(1, 256**3),
                         xl[1].reshape(1, 256**3),
                         xl[2].reshape(1, 256**3),
                         255 * np.ones((1, 256**3)))).T
        return lut.astype('int32')
    __lut__ = __create_8bit_rgb_lut__()

    @staticmethod
    def rgb2scalar(rgb):
        """
        return index of RGB colors into the LUT table.
        rgb : nx3 array

        вернуть индекс цветов RGB в таблицу LUT.
         RGB: массив NX3
        """
        return 256**2*rgb[:,0] + 256*rgb[:,1] + rgb[:,2]

    @staticmethod
    def set_rgb_lut(myv_glyph):
        """
        Sets the LUT of the Mayavi glyph to RGB-LUT.

        Устанавливает LUT глифа майави в RGB-LUT.
        """
        lut = myv_glyph.module_manager.scalar_lut_manager.lut
        lut._vtk_obj.SetTableRange(0, LUT_RGB.__lut__.shape[0])
        lut.number_of_colors = LUT_RGB.__lut__.shape[0]
        lut.table = LUT_RGB.__lut__


def plot_xyzrgb(xyz,rgb,show=False):
    """
    xyz : nx3 float
    rgb : nx3 uint8

    Plots a RGB-D point-cloud in mayavi.

    xyz: nx3 float
    rgb: nx3 uint8

    Создает облако точек RGB-D в Mayavi.
    """
    rgb_s = LUT_RGB.rgb2scalar(rgb)
    pts_glyph = mym.points3d(xyz[:,0],xyz[:,1],xyz[:,2],
                             rgb_s,mode='point')
    LUT_RGB.set_rgb_lut(pts_glyph)
    if show:
        mym.view(180,180)
        mym.orientation_axes()
        mym.show()

def visualize_plane(pt,plane,show=False):
    """
    Visualize the RANSAC PLANE (4-tuple) fit to PT (nx3 array).
    Also draws teh normal.

    Визуализируйте посадку RANSAC PLANE (4 кортежа) в PT (массив nx3).
    Также рисует нормально.
    """
    # # plot the point-cloud:
    # # построим облако точек:
    if show and mym.gcf():
        mym.clf()

    # plot the plane:
    # построить самолет:
    plane_eq = '%f*x+%f*y+%f*z+%f'%tuple(plane.tolist())
    m,M = np.percentile(pt,[10,90],axis=0)
    implicit_plot(plane_eq, (m[0],M[0],m[1],M[1],m[2],M[2]))

    # plot surface normal:
    # нормальная поверхность графика:
    mu = np.percentile(pt,50,axis=0)
    mu_z = -(mu[0]*plane[0]+mu[1]*plane[1]+plane[3])/plane[2]
    mym.quiver3d(mu[0],mu[1],mu_z,plane[0],plane[1],plane[2],scale_factor=0.3)

    if show:
        mym.view(180,180)
        mym.orientation_axes()
        mym.show(True)

def implicit_plot(expr, ext_grid, Nx=11, Ny=11, Nz=11,
                 col_isurf=(50/255, 199/255, 152/255)):
    """
    Function to plot algebraic surfaces described by implicit equations in Mayavi
 
    Implicit functions are functions of the form
 
        `F(x,y,z) = c`
 
    where `c` is an arbitrary constant.
 
    Parameters
    ----------
    expr : string
        The expression `F(x,y,z) - c`; e.g. to plot a unit sphere,
        the `expr` will be `x**2 + y**2 + z**2 - 1`
    ext_grid : 6-tuple
        Tuple denoting the range of `x`, `y` and `z` for grid; it has the
        form - (xmin, xmax, ymin, ymax, zmin, zmax)
    fig_handle : figure handle (optional)
        If a mayavi figure object is passed, then the surface shall be added
        to the scene in the given figure. Then, it is the responsibility of
        the calling function to call mlab.show().
    Nx, Ny, Nz : Integers (optional, preferably odd integers)
        Number of points along each axis. It is recommended to use odd numbers
        to ensure the calculation of the function at the origin.

        Функция для построения алгебраических поверхностей, описываемых неявными уравнениями в Mayavi
 
     Неявные функции являются функциями вида
 
         `F (x, y, z) = c`
 
     где `c` - произвольная постоянная.
 
     параметры
     ----------
     expr: строка
         Выражение `F (x, y, z) - c`; например построить единичную сферу,
         `expr` будет` x ** 2 + y ** 2 + z ** 2 - 1`
     ext_grid: 6-кортеж
         Кортеж, обозначающий диапазон `x`,` y` и `z` для сетки; это имеет
         форма - (xmin, xmax, ymin, ymax, zmin, zmax)
     fig_handle: дескриптор фигуры (необязательно)
         Если объект фигуры майави пропущен, то поверхность должна быть добавлена
         на сцену в данной фигуре. Тогда это ответственность
         вызывающая функция для вызова mlab.show ().
     Nx, Ny, Nz: целые числа (необязательно, предпочтительно нечетные целые числа)
         Количество точек по каждой оси. Рекомендуется использовать нечетные числа
         обеспечить расчет функции в начале координат.
    """
    xl, xr, yl, yr, zl, zr = ext_grid
    x, y, z = np.mgrid[xl:xr:eval('{}j'.format(Nx)),
                       yl:yr:eval('{}j'.format(Ny)),
                       zl:zr:eval('{}j'.format(Nz))]
    scalars = eval(expr)
    src = mym.pipeline.scalar_field(x, y, z, scalars)
    cont1 = mym.pipeline.iso_surface(src, color=col_isurf, contours=[0],
                                      transparent=False, opacity=0.8)
    cont1.compute_normals = False # for some reasons, setting this to true actually cause
                                  # more unevenness on the surface, instead of more smooth
                                    # по некоторым причинам, установка этого значения в истинное вызывает
                                    # больше неровностей на поверхности, а не более гладких
    cont1.actor.property.specular = 0.2 #0.4 #0.8
    cont1.actor.property.specular_power = 55.0 #15.0


def ensure_proj_z(plane_coeffs, min_z_proj):
    a,b,c,d = plane_coeffs
    if np.abs(c) < min_z_proj:
        s = ((1 - min_z_proj**2) / (a**2 + b**2))**0.5
        coeffs = np.array([s*a, s*b, np.sign(c)*min_z_proj, d])
        assert np.abs(np.linalg.norm(coeffs[:3])-1) < 1e-3
        return coeffs
    return plane_coeffs

def isplanar(xyz,sample_neighbors,dist_thresh,num_inliers,z_proj):
    """
    Checks if at-least FRAC_INLIERS fraction of points of XYZ (nx3)
    points lie on a plane. The plane is fit using RANSAC.

    XYZ : (nx3) array of 3D point coordinates
    SAMPLE_NEIGHBORS : 5xN_RANSAC_TRIALS neighbourhood array
                       of indices into the XYZ array. i.e. the values in this
                       matrix range from 0 to number of points in XYZ
    DIST_THRESH (default = 10cm): a point pt is an inlier iff dist(plane-pt)<dist_thresh
    FRAC_INLIERS : fraction of total-points which should be inliers to
                   to declare that points are planar.
    Z_PROJ : changes the surface normal, so that its projection on z axis is ATLEAST z_proj.

    Returns:
        None, if the data is not planar, else a 4-tuple of plane coeffs.

        Проверяет, является ли FRAC_INLIERS хотя бы долей точек XYZ (nx3)
     точки лежат на плоскости. Самолет подгоняется с помощью RANSAC.

     XYZ: (nx3) массив координат трехмерной точки
     SAMPLE_NEIGHBORS: 5xN_RANSAC_TRIALS массив окрестностей
                        индексов в массив XYZ. то есть значения в этом
                        диапазон матрицы от 0 до количества точек в XYZ
     DIST_THRESH (по умолчанию = 10 см): точка pt является внутренним, если dist (plane-pt) <dist_thresh
     FRAC_INLIERS: доля от общего количества баллов, которые должны быть
                    объявить, что точки являются плоскими.
     Z_PROJ: изменяет нормаль поверхности, так что ее проекция на ось z равна ATLEAST z_proj.

     Возвращает:
         Нет, если данные не плоские, иначе 4 кортежа плоских коэффициентов.
    """
    frac_inliers = num_inliers/xyz.shape[0]
    dv = -np.percentile(xyz,50,axis=0) # align the normal to face towards camera
    max_iter = sample_neighbors.shape[-1]
    plane_info =  fit_plane_ransac(xyz,neighbors=sample_neighbors,
                            z_pos=dv,dist_inlier=dist_thresh,
                            min_inlier_frac=frac_inliers,nsample=20,
                            max_iter=max_iter) 
    if plane_info != None:
        coeff, inliers = plane_info
        coeff = ensure_proj_z(coeff, z_proj)
        return coeff,inliers
    else:
        return #None


class DepthCamera(object):
    """
    Camera functions for Depth-CNN camera.

    Функции камеры для камеры Depth-CNN.
    """
    f = 520

    @staticmethod
    def plane2xyz(center, ij, plane):
        """
        converts image pixel indices to xyz on the PLANE.

        center : 2-tuple
        ij : nx2 int array
        plane : 4-tuple

        return nx3 array.

        преобразует пиксельные индексы изображения в плоскость xyz.

         центр: 2 кортежа
         ij: массив nx2 int
         плоскость: 4-кортеж

         вернуть массив nx3.
        """
        ij = np.atleast_2d(ij)
        n = ij.shape[0]
        ij = ij.astype('float')
        xy_ray = (ij-center[None,:]) / DepthCamera.f
        z = -plane[2]/(xy_ray.dot(plane[:2])+plane[3])
        xyz = np.c_[xy_ray, np.ones(n)] * z[:,None]
        return xyz

    @staticmethod
    def depth2xyz(depth):
        """
        Convert a HxW depth image (float, in meters)
        to XYZ (HxWx3).

        y is along the height.
        x is along the width.

        Преобразование изображения глубиной HxW (с плавающей точкой, в метрах)
         в XYZ (HxWx3).

         у вдоль высоты.
         х по ширине.
        """
        H,W = depth.shape
        xx,yy = np.meshgrid(np.arange(W),np.arange(H))
        X = (xx-W/2) * depth / DepthCamera.f
        Y = (yy-H/2) * depth / DepthCamera.f
        return np.dstack([X,Y,depth.copy()])

    @staticmethod
    def overlay(rgb,depth):
        """
        overlay depth and rgb images for visualization:

        Наложение глубины и RGB изображений для визуализации:
        """
        depth = depth - np.min(depth)
        depth /= np.max(depth)
        depth = (255*depth).astype('uint8')
        return np.dstack([rgb[:,:,0],depth,rgb[:,:,1]])


def get_texture_score(img,masks,labels):
    """
    gives a textureness-score
    (low -> less texture, high -> more texture) for each mask.

    дает оценку текстуры
     (низкая -> меньше текстуры, высокая -> больше текстуры) для каждой маски.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.astype('float32')/255.0
    img = sim.filters.gaussian_filter(img,sigma=1)  
    G = np.clip(np.abs(sim.filters.laplace(img)),0,1)

    tex_score = []
    for l in labels:
        ts = np.sum(G[masks==l].flat)/np.sum((masks==l).flat)
        tex_score.append(ts)
    return np.array(tex_score)

def ssc(v):
    """
    Returns the skew-symmetric cross-product matrix corresponding to v.

    Возвращает кососимметричную матрицу кросс-произведения, соответствующую v.
    """
    v /= np.linalg.norm(v)
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def rot3d(v1,v2):
    """
    Rodrigues formula : find R_3x3 rotation matrix such that v2 = R*v1.
    https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation

    Формула Родрига: найдите матрицу вращения R_3x3 такую, что v2 = R * v1.
     https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1,v2)
    s = np.linalg.norm(v3)
    c = v1.dot(v2)
    Vx = ssc(v3)
    return np.eye(3)+s*Vx+(1-c)*Vx.dot(Vx)

def unrotate2d(pts):
    """
    PTS : nx3 array
    finds principal axes of pts and gives a rotation matrix (2d)
    to realign the axes of max variance to x,y.

    PTS: массив nx3
     находит главные оси точек и дает матрицу вращения (2d)
     выровнять оси максимальной дисперсии по x, y.
    """
    mu = np.median(pts,axis=0)
    pts -= mu[None,:]
    l,R = np.linalg.eig(pts.T.dot(pts))
    R = R / np.linalg.norm(R,axis=0)[None,:]

    # make R compatible with x-y axes:
    if abs(R[0,0]) < abs(R[0,1]): #compare dot-products with [1,0].T
        R = np.fliplr(R)
    if not np.allclose(np.linalg.det(R),1):
        if R[0,0]<0:
            R[:,0] *= -1
        elif R[1,1]<0:
            R[:,1] *= -1
        else:
            print ("Rotation matrix not understood")
            return
    if R[0,0]<0 and R[1,1]<0:
        R *= -1
    assert np.allclose(np.linalg.det(R),1)

    # at this point "R" is a basis for the original (rotated) points.
    # we need to return the inverse to "unrotate" the points:
    # в этой точке «R» является основой для исходных (повернутых) точек.
    # нам нужно вернуть обратное, чтобы «развернуть» точки:
    return R.T #return the inverse
    # вернуть обратное
