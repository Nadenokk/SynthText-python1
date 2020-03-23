# Author: Ankush Gupta
# Date: 2015

"""
Main script for synthetic text rendering.

Основной скрипт для рендеринга синтетического текста.
"""

from __future__ import division
import copy
import cv2
import h5py
from PIL import Image
import numpy as np 
#import mayavi.mlab as mym
import matplotlib.pyplot as plt 
import os.path as osp
import scipy.ndimage as sim
import scipy.spatial.distance as ssd
import synth_utils as su
import text_utils as tu
from colorize3_poisson import Colorize
from common import *
import traceback, itertools


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.

    Получить регион из сегментации, которые хороши для размещения
    текст.
    """
    minWidth = 30 #px
    minHeight = 30 #px
    minAspect = 0.3 # w > 0.3*h
    maxAspect = 7
    minArea = 100 # number of pix
    # количество пикселей
    pArea = 0.60 # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    # Планарные фитинги RANSAC:
    dist_thresh = 0.10 # m
    num_inlier = 90
    ransac_fit_trials = 100
    min_z_projection = 0.25

    minW = 20

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"

        маска: 1 где «ВКЛ», 0 где «ВЫКЛ»
        """
        wx = np.median(np.sum(mask,axis=0))
        wy = np.median(np.sum(mask,axis=1))
        return wx>TextRegions.minW and wy>TextRegions.minW

    @staticmethod
    def get_hw(pt,return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt,axis=0)
        pt = (pt-mu[None,:]).dot(R.T) + mu[None,:]
        h,w = np.max(pt,axis=0) - np.min(pt,axis=0)
        if return_rot:
            return h,w,R
        return h,w
 
    @staticmethod
    def filter(seg,area,label):
        """
        Apply the filter.
        The final list is ranked by area.

        Примените фильтр.
        Финальный список ранжируется по областям.
        """
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt,R = [],[]
        for idx,i in enumerate(good):
            mask = seg==i
            xs,ys = np.where(mask)

            coords = np.c_[xs,ys].astype('float32')
            rect = cv2.minAreaRect(coords)          
            #box = np.array(cv2.cv.BoxPoints(rect))
            box = np.array(cv2.boxPoints(rect))
            h,w,rot = TextRegions.get_hw(box,return_rot=True)

            f = (h > TextRegions.minHeight 
                and w > TextRegions.minWidth
                and TextRegions.minAspect < w/h < TextRegions.maxAspect
                and area[idx]/w*h > TextRegions.pArea)
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        # фильтровать плохие регионы:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        # сортировать регионы по областям:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label':good, 'rot':R, 'area': area[aidx]}
        return filter_info

    @staticmethod
    def sample_grid_neighbours(mask,nsample,step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.

        Учитывая двоичную маску HxW, образец 4 соседей по сетке,
        в кардинальных направлениях, шаг за шагом пикселей.
        """
        if 2*step >= min(mask.shape[:2]):
            return #None

        y_m,x_m = np.where(mask)
        mask_idx = np.zeros_like(mask,'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i],x_m[i]] = i

        xp,xn = np.zeros_like(mask), np.zeros_like(mask)
        yp,yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:,:-2*step] = mask[:,2*step:]
        xn[:,2*step:] = mask[:,:-2*step]
        yp[:-2*step,:] = mask[2*step:,:]
        yn[2*step:,:] = mask[:-2*step,:]
        valid = mask&xp&xn&yp&yn

        ys,xs = np.where(valid)
        N = len(ys)
        if N==0: #no valid pixels in mask:
            # нет действительных пикселей в маске:
            return #None
        # Никто
        nsample = min(nsample,N)
        idx = np.random.choice(N,nsample,replace=False)
        # generate neighborhood matrix:
        # сгенерировать матрицу окрестностей:
        # (1+4)x2xNsample (2 for y,x)
        xs,ys = xs[idx],ys[idx]
        s = step
        X = np.transpose(np.c_[xs,xs+s,xs+s,xs-s,xs-s][:,:,None],(1,2,0))
        Y = np.transpose(np.c_[ys,ys+s,ys-s,ys+s,ys-s][:,:,None],(1,2,0))
        sample_idx = np.concatenate([Y,X],axis=1)
        mask_nn_idx = np.zeros((5,sample_idx.shape[-1]),'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:,i] = mask_idx[sample_idx[:,:,i][:,0],sample_idx[:,:,i][:,1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz,seg,regions):
        plane_info = {'label':[],
                      'coeff':[],
                      'support':[],
                      'rot':[],
                      'area':[]}
        for idx,l in enumerate(regions['label']):
            mask = seg==l
            pt_sample = TextRegions.sample_grid_neighbours(mask,TextRegions.ransac_fit_trials,step=3)
            if pt_sample is None:
                continue #not enough points for RANSAC
                # не хватает очков для RANSAC
            # get-depths
            # получить глубины
            pt = xyz[mask]
            plane_model = su.isplanar(pt, pt_sample,
                                     TextRegions.dist_thresh,
                                     TextRegions.num_inlier,
                                     TextRegions.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2])>TextRegions.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

        return plane_info

    @staticmethod
    def get_regions(xyz,seg,area,label):
        regions = TextRegions.filter(seg,area,label)
        # fit plane to text-regions:
        regions = TextRegions.filter_depth(xyz,seg,regions)
        return regions

def rescale_frontoparallel(p_fp,box_fp,p_im):
    """
    The fronto-parallel image region is rescaled to bring it in 
    the same approx. size as the target region size.

    p_fp : nx2 coordinates of countour points in the fronto-parallel plane
    box  : 4x2 coordinates of bounding box of p_fp
    p_im : nx2 coordinates of countour in the image

    NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

    Returns the scale 's' to scale the fronto-parallel points by.

    Фронтопараллельная область изображения масштабируется, чтобы привести ее в
     тот же ок. размер как размер целевой области.

     p_fp: nx2 координаты точек отсчета во фронтально-параллельной плоскости
     box: 4x2 координаты ограничительной рамки p_fp
     p_im: nx2 координаты отсчета на изображении

     ПРИМЕЧАНИЕ: p_fp и p соответствуют, то есть: p_fp [i] ~ p [i]

     Возвращает шкалу 's' для масштабирования фронто-параллельных точек.

    """
    l1 = np.linalg.norm(box_fp[1,:]-box_fp[0,:])
    l2 = np.linalg.norm(box_fp[1,:]-box_fp[2,:])

    n0 = np.argmin(np.linalg.norm(p_fp-box_fp[0,:][None,:],axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp-box_fp[1,:][None,:],axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp-box_fp[2,:][None,:],axis=1))

    lt1 = np.linalg.norm(p_im[n1,:]-p_im[n0,:])
    lt2 = np.linalg.norm(p_im[n1,:]-p_im[n2,:])

    s =  max(lt1/l1,lt2/l2)
    if not np.isfinite(s):
        s = 1.0
    return s

def get_text_placement_mask(xyz,mask,plane,pad=2,viz=False):
    """
    Returns a binary mask in which text can be placed.
    Also returns a homography from original image
    to this rectified mask.

    XYZ  : (HxWx3) image xyz coordinates
    MASK : (HxW) : non-zero pixels mark the object mask
    REGION : DICT output of TextRegions.get_regions
    PAD : number of pixels to pad the placement-mask by

    Возвращает двоичную маску, в которую можно поместить текст.
     Также возвращает гомографию из исходного изображения
     к этой выпрямленной маске.

     XYZ: (HxWx3) координаты xyz изображения
     MASK: (HxW): ненулевые пиксели отмечают маску объекта
     REGION: DICT-вывод TextRegions.get_regions
     PAD: количество пикселей для заполнения маски размещения
    """
    _,contour,hier = cv2.findContours(mask.copy().astype('uint8'),
                                    mode=cv2.RETR_CCOMP,
                                    method=cv2.CHAIN_APPROX_SIMPLE)
    contour = [np.squeeze(c).astype('float') for c in contour]
    #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
    H,W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    # перенести контурные точки в параллелепрограмму:
    pts,pts_fp = [],[]
    center = np.array([W,H])/2
    n_front = np.array([0.0,0.0,-1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = su.DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = su.rot3d(plane[:3],n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:,:2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    # развернуть в 2D плоскости:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = su.unrotate2d(box.copy())
    box = np.vstack([box,box[0,:]]) #close the box for visualization
    # закрыть окно для визуализации

    mu = np.median(pts_fp[0],axis=0)
    pts_tmp = (pts_fp[0]-mu[None,:]).dot(R2d.T) + mu[None,:]
    boxR = (box-mu[None,:]).dot(R2d.T) + mu[None,:]
    
    # rescale the unrotated 2d points to approximately
    # the same scale as the target region:
    # масштабировать необращенные 2d точки примерно
    # в том же масштабе, что и целевой регион:
    s = rescale_frontoparallel(pts_tmp,boxR,pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s*((pts_fp[i]-mu[None,:]).dot(R2d.T) + mu[None,:])

    # paint the unrotated contour points:
    # закрасить не повернутые точки контура:
    minxy = -np.min(boxR,axis=0) + pad//2
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:,0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:,1]).T))

    place_mask = 255*np.ones((int(np.ceil(COL))+pad, int(np.ceil(ROW))+pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i]+minxy[None,:]).astype('int32') for i in range(len(pts_fp))]
    cv2.drawContours(place_mask,pts_fp_i32,-1,0,
                     thickness=cv2.FILLED,
                     lineType=8,hierarchy=hier)
    
    if not TextRegions.filter_rectified((~place_mask).astype('float')/255):
        return

    # calculate the homography
    # рассчитать гомографию
    H,_ = cv2.findHomography(pts[0].astype('float32').copy(),
                             pts_fp_i32[0].astype('float32').copy(),
                             method=0)

    Hinv,_ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                pts[0].astype('float32').copy(),
                                method=0)
    if viz:
        plt.subplot(1,2,1)
        plt.imshow(mask)
        plt.subplot(1,2,2)
        plt.imshow(~place_mask)
        plt.hold(True)
        for i in range(len(pts_fp_i32)):
            plt.scatter(pts_fp_i32[i][:,0],pts_fp_i32[i][:,1],
                        edgecolors='none',facecolor='g',alpha=0.5)
        plt.show()

    return place_mask,H,Hinv

def viz_masks(fignum,rgb,seg,depth,label):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.

    img, глубина, сег - это изображения одинакового размера.
    визуализирует маски глубины для лучших объектов NOBJ.
    """
    def mean_seg(rgb,seg,label):
        mim = np.zeros_like(rgb)
        for i in np.unique(seg.flat):
            mask = seg==i
            col = np.mean(rgb[mask,:],axis=0)
            mim[mask,:] = col[None,None,:]
        mim[seg==0,:] = 0
        return mim

    mim = mean_seg(rgb,seg,label)

    img = rgb.copy()
    for i,idx in enumerate(label):
        mask = seg==idx
        rgb_rand = (255*np.random.rand(3)).astype('uint8')
        img[mask] = rgb_rand[None,None,:] 

    #import scipy
    # scipy.misc.imsave('seg.png', mim)
    # scipy.misc.imsave('depth.png', depth)
    # scipy.misc.imsave('txt.png', rgb)
    # scipy.misc.imsave('reg.png', img)

    plt.close(fignum)
    plt.figure(fignum)
    ims = [rgb,mim,depth,img]
    for i in range(len(ims)):
        plt.subplot(2,2,i+1)
        plt.imshow(ims[i])
    plt.show(block=False)

def viz_regions(img,xyz,seg,planes,labels):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.

    img, глубина, сег - это изображения одинакового размера.
    визуализирует маски глубины для лучших объектов NOBJ.
    """
    # plot the RGB-D point-cloud:
    # построить облако точек RGB-D:
    su.plot_xyzrgb(xyz.reshape(-1,3),img.reshape(-1,3))

    # plot the RANSAC-planes at the text-regions:
    # построить RANSAC-плоскости в текстовых областях:
    for i,l in enumerate(labels):
        mask = seg==l
        xyz_region = xyz[mask,:]
        su.visualize_plane(xyz_region,np.array(planes[i]))

    mym.view(180,180)
    mym.orientation_axes()
    mym.show(True)
 
def viz_textbb(fignum,text_im, bb_list,alpha=1.0):
    """
    text_im : image containing text
    bb_list : list of 2x4xn_i boundinb-box matrices

    text_im: изображение, содержащее текст
    bb_list: список матриц 2x4xn_i границ
    """
    plt.close(fignum)
    plt.figure(fignum)
    plt.imshow(text_im)
    plt.hold(True)
    H,W = text_im.shape[:2]
    for i in range(len(bb_list)):
        bbs = bb_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

class RendererV3(object):

    def __init__(self, data_dir, max_time=None):
        self.text_renderer = tu.RenderFont(data_dir)
        self.colorizer = Colorize(data_dir)
        #self.colorizerV2 = colorV2.Colorize(data_dir)

        self.min_char_height = 8 #px
        self.min_asp_ratio = 0.4 #

        self.max_text_regions = 7

        self.max_time = max_time

    def filter_regions(self,regions,filt):
        """
        filt : boolean list of regions to keep.

        Filta: логический список регионов, которые нужно сохранить.
        """
        idx = np.arange(len(filt))[filt]
        for k in regions.keys():
            regions[k] = [regions[k][i] for i in idx]
        return regions

    def filter_for_placement(self,xyz,seg,regions):
        filt = np.zeros(len(regions['label'])).astype('bool')
        masks,Hs,Hinvs = [],[], []
        for idx,l in enumerate(regions['label']):
            res = get_text_placement_mask(xyz,seg==l,regions['coeff'][idx],pad=2)
            if res is not None:
                mask,H,Hinv = res
                masks.append(mask)
                Hs.append(H)
                Hinvs.append(Hinv)
                filt[idx] = True
        regions = self.filter_regions(regions,filt)
        regions['place_mask'] = masks
        regions['homography'] = Hs
        regions['homography_inv'] = Hinvs

        return regions

    def warpHomography(self,src_mat,H,dst_size):
        dst_mat = cv2.warpPerspective(src_mat, H, dst_size,
                                      flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        return dst_mat

    def homographyBB(self, bbs, H, offset=None):
        """
        Apply homography transform to bounding-boxes.
        BBS: 2 x 4 x n matrix  (2 coordinates, 4 points, n bbs).
        Returns the transformed 2x4xn bb-array.

        offset : a 2-tuple (dx,dy), added to points before transfomation.

        Примените преобразование гомографии к ограничивающим прямоугольникам.
         BBS: матрица 2 x 4 x n (2 координаты, 4 точки, n bbs).
         Возвращает преобразованный 2x4xn bb-массив.

         смещение: 2-кортеж (dx, dy), добавленный к точкам перед трансфомацией.
        """
        eps = 1e-16
        # check the shape of the BB array:
        # проверить форму массива BB:
        t,f,n = bbs.shape
        assert (t==2) and (f==4)

        # append 1 for homogenous coordinates:
        # добавить 1 для однородных координат:
        bbs_h = np.reshape(np.r_[bbs, np.ones((1,4,n))], (3,4*n), order='F')
        if offset != None:
            bbs_h[:2,:] += np.array(offset)[:,None]

        # perpective:
        # перспектива:
        bbs_h = H.dot(bbs_h)
        bbs_h /= (bbs_h[2,:]+eps)

        bbs_h = np.reshape(bbs_h, (3,4,n), order='F')
        return bbs_h[:2,:,:]

    def bb_filter(self,bb0,bb,text):
        """
        Ensure that bounding-boxes are not too distorted
        after perspective distortion.

        bb0 : 2x4xn martrix of BB coordinates before perspective
        bb  : 2x4xn matrix of BB after perspective
        text: string of text -- for excluding symbols/punctuations.

        Убедитесь, что ограничивающие рамки не слишком искажены
         после искажения перспективы.

         bb0: 2x4xn мартрикс координат BB перед перспективой
         bb: матрица BB 2x4xn после перспективы
         text: строка текста - для исключения символов / знаков препинания.
        """
        h0 = np.linalg.norm(bb0[:,3,:] - bb0[:,0,:], axis=0)
        w0 = np.linalg.norm(bb0[:,1,:] - bb0[:,0,:], axis=0)
        hw0 = np.c_[h0,w0]

        h = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        w = np.linalg.norm(bb[:,1,:] - bb[:,0,:], axis=0)
        hw = np.c_[h,w]

        # remove newlines and spaces:
        # удалить переводы строки и пробелы:
        text = ''.join(text.split())
        assert len(text)==bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        hw0 = hw0[alnum,:]
        hw = hw[alnum,:]

        min_h0, min_h = np.min(hw0[:,0]), np.min(hw[:,0])
        asp0, asp = hw0[:,0]/hw0[:,1], hw[:,0]/hw[:,1]
        asp0, asp = np.median(asp0), np.median(asp)

        asp_ratio = asp/asp0
        is_good = ( min_h > self.min_char_height
                    and asp_ratio > self.min_asp_ratio
                    and asp_ratio < 1.0/self.min_asp_ratio)
        return is_good


    def get_min_h(selg, bb, text):
        # find min-height:
        h = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        # remove newlines and spaces:
        # удалить переводы строки и пробелы:
        text = ''.join(text.split())
        assert len(text)==bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        h = h[alnum]
        return np.min(h)


    def feather(self, text_mask, min_h):
        # determine the gaussian-blur std:
        # определить стандарт gaussian-blur
        if min_h <= 15 :
            bsz = 0.25
            ksz=1
        elif 15 < min_h < 30:
            bsz = max(0.30, 0.5 + 0.1*np.random.randn())
            ksz = 3
        else:
            bsz = max(0.5, 1.5 + 0.5*np.random.randn())
            ksz = 5
        return cv2.GaussianBlur(text_mask,(ksz,ksz),bsz)

    def place_text(self,rgb,collision_mask,H,Hinv):
        font = self.text_renderer.font_state.sample()
        font = self.text_renderer.font_state.init_font(font)

        render_res = self.text_renderer.render_sample(font,collision_mask)
        if render_res is None: # rendering not successful
            # рендеринг не успешен
            return #None
            # Никто
        else:
            text_mask,loc,bb,text = render_res

        # update the collision mask with text:
        # обновить маску столкновения с текстом:
        collision_mask += (255 * (text_mask>0)).astype('uint8')

        # warp the object mask back onto the image:
        # перенести маску объекта обратно на изображение:
        text_mask_orig = text_mask.copy()
        bb_orig = bb.copy()
        text_mask = self.warpHomography(text_mask,H,rgb.shape[:2][::-1])
        bb = self.homographyBB(bb,Hinv)

        if not self.bb_filter(bb_orig,bb,text):
            #warn("bad charBB statistics")
            return #None

        # get the minimum height of the character-BB:
        # получить минимальную высоту символа-ВВ:
        min_h = self.get_min_h(bb,text)

        #feathering:
        # оперение:
        text_mask = self.feather(text_mask, min_h)

        im_final = self.colorizer.color(rgb,[text_mask],np.array([min_h]))

        return im_final, text, bb, collision_mask


    def get_num_text_regions(self, nregions):
        #return nregions
        # вернуть регионы
        nmax = min(self.max_text_regions, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0,1.0)
        return int(np.ceil(nmax * rnd))

    def char2wordBB(self, charBB, text):
        """
        Converts character bounding-boxes to word-level
        bounding-boxes.

        charBB : 2x4xn matrix of BB coordinates
        text   : the text string

        output : 2x4xm matrix of BB coordinates,
                 where, m == number of words.

        Преобразует ограничивающие рамки символов в уровень слова
         Ограничивающие-боксы.

         charBB: матрица 2x4xn координат BB
         текст: текстовая строка

         вывод: 2x4xm матрица координат BB,
                  где, м == количество слов.
        """
        wrds = text.split()
        bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
        wordBB = np.zeros((2,4,len(wrds)), 'float32')
        
        for i in range(len(wrds)):
            cc = charBB[:,:,bb_idx[i]:bb_idx[i+1]]

            # fit a rotated-rectangle:
            # change shape from 2x4xn_i -> (4*n_i)x2
            # соответствовать повернутому прямоугольнику:
            # изменить форму с 2x4xn_i -> (4 * n_i) x2
            cc = np.squeeze(np.concatenate(np.dsplit(cc,cc.shape[-1]),axis=1)).T.astype('float32')
            rect = cv2.minAreaRect(cc.copy())
            box = np.array(cv2.boxPoints(rect))

            # find the permutation of box-coordinates which
            # are "aligned" appropriately with the character-bb.
            # (exhaustive search over all possible assignments):
            # найти перестановку коробчатых координат, которая
            # "выровнены" соответствующим образом с символом -bb.
            # (исчерпывающий поиск по всем возможным заданиям):
            cc_tblr = np.c_[cc[0,:],
                            cc[-3,:],
                            cc[-2,:],
                            cc[3,:]].T
            perm4 = np.array(list(itertools.permutations(np.arange(4))))
            dists = []
            for pidx in range(perm4.shape[0]):
                d = np.sum(np.linalg.norm(box[perm4[pidx],:]-cc_tblr,axis=1))
                dists.append(d)
            wordBB[:,:,i] = box[perm4[np.argmin(dists)],:].T

        return wordBB


    def render_text(self,rgb,depth,seg,area,label,ninstance=1,viz=False):
        """
        rgb   : HxWx3 image rgb values (uint8)
        depth : HxW depth values (float)
        seg   : HxW segmentation region masks
        area  : number of pixels in each region
        label : region labels == unique(seg) / {0}
               i.e., indices of pixels in SEG which
               constitute a region mask
        ninstance : no of times image should be
                    used to place text.

        @return:
            res : a list of dictionaries, one for each of 
                  the image instances.
                  Each dictionary has the following structure:
                      'img' : rgb-image with text on it.
                      'bb'  : 2x4xn matrix of bounding-boxes
                              for each character in the image.
                      'txt' : a list of strings.

                  The correspondence b/w bb and txt is that
                  i-th non-space white-character in txt is at bb[:,:,i].
            
            If there's an error in pre-text placement, for e.g. if there's 
            no suitable region for text placement, an empty list is returned.

            RGB: HxWx3 изображения RGB значения (Uint8)
        глубина: значения глубины HxW (с плавающей точкой)
        seg: маски области сегментации HxW
        area: количество пикселей в каждой области
        метка: метки региона == уникальный (сегмент) / {0}
               то есть индексы пикселей в SEG, которые
               составляют маску региона
        Ninstance: не раз изображение должно быть
                    используется для размещения текста.

        @возвращение:
            res: список словарей, по одному для каждого из
                  экземпляры изображения.
                  Каждый словарь имеет следующую структуру:
                      'img': rgb-изображение с текстом на нем.
                      'bb': матрица 2x4xn ограничивающих прямоугольников
                              для каждого символа в изображении.
                      'txt': список строк.

                  Соответствие ч / б bb и txt таково, что
                  i-й непробельный белый символ в txt находится на bb [:,:, i].
            
            Если есть ошибка в размещении пре-текста, например, если есть
            нет подходящего региона для размещения текста, возвращается пустой список.
        """
        try:
            # depth -> xyz
            # глубина -> XYZ
            xyz = su.DepthCamera.depth2xyz(depth)
            
            # find text-regions:
            # найти текстовые регионы:
            regions = TextRegions.get_regions(xyz,seg,area,label)

            # find the placement mask and homographies:
            # найти маску размещения и омографии:
            regions = self.filter_for_placement(xyz,seg,regions)

            # finally place some text:
            # наконец поместите текст:
            nregions = len(regions['place_mask'])
            if nregions < 1: # no good region to place text on
                # нет хорошего региона для размещения текста
                return []
        except:
            # failure in pre-text placement
            # ошибка при размещении пре-текста
            #import traceback
            traceback.print_exc()
            return []

        res = []
        for i in range(ninstance):
            place_masks = copy.deepcopy(regions['place_mask'])

            print (colorize(Color.CYAN, " ** instance # : %d"%i))

            idict = {'img':[], 'charBB':None, 'wordBB':None, 'txt':None}

            m = self.get_num_text_regions(nregions)#np.arange(nregions)#min(nregions, 5*ninstance*self.max_text_regions))
            reg_idx = np.arange(min(2*m,nregions))
            np.random.shuffle(reg_idx)
            reg_idx = reg_idx[:m]

            placed = False
            img = rgb.copy()
            itext = []
            ibb = []

            # process regions:
            # области обработки:
            num_txt_regions = len(reg_idx)
            NUM_REP = 5 # re-use each region three times:
            # повторно использовать каждый регион три раза:
            reg_range = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
            for idx in reg_range:
                ireg = reg_idx[idx]
                try:
                    if self.max_time is None:
                        txt_render_res = self.place_text(img,place_masks[ireg],
                                                         regions['homography'][ireg],
                                                         regions['homography_inv'][ireg])
                    else:
                        with time_limit(self.max_time):
                            txt_render_res = self.place_text(img,place_masks[ireg],
                                                             regions['homography'][ireg],
                                                             regions['homography_inv'][ireg])
                except TimeoutException as msg:
                    print (msg)
                    continue
                except:
                    traceback.print_exc()
                    # some error in placing text on the region
                    # ошибка при размещении текста в регионе
                    continue

                if txt_render_res is not None:
                    placed = True
                    img,text,bb,collision_mask = txt_render_res
                    # update the region collision mask:
                    # обновить маску коллизий региона:
                    place_masks[ireg] = collision_mask
                    # store the result:
                    # сохранить результат:
                    itext.append(text)
                    ibb.append(bb)

            if  placed:
                # at least 1 word was placed in this instance:
                # как минимум 1 слово было помещено в этот экземпляр:
                idict['img'] = img
                idict['txt'] = itext
                idict['charBB'] = np.concatenate(ibb, axis=2)
                idict['wordBB'] = self.char2wordBB(idict['charBB'].copy(), ' '.join(itext))
                res.append(idict.copy())
                if viz:
                    viz_textbb(1,img, [idict['wordBB']], alpha=1.0)
                    viz_masks(2,img,seg,depth,regions['label'])
                    # viz_regions(rgb.copy(),xyz,seg,regions['coeff'],regions['label'])
                    if i < ninstance-1:
                        raw_input(colorize(Color.BLUE,'continue?',True))                    
        return res
