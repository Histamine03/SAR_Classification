#wavelet_edit.py의 경우 라이브러리 scikit-image(https://scikit-image.org/)의 
#‘_wavelet_threshold’ 함수의 소스코드를 수정하여 이용하였다. 
# 기존의 _wavelet_threshold함수의 threshold 추정 방법 ‘VisuShrink’, ‘BayesShrink’ 외에 
# 2가지의 방식을 'Fuzzy', 'new_Visu'라는 이름으로 추가하였다.
# _fuzzy_thresh함수는 이를 위해 생성한 함수이다.

import scipy.stats
import pywt
from warnings import warn

def _universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """ #sigma는 cD1에서 추정된 값.
    return sigma*np.sqrt(2*np.log(img.size))


def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def _fuzzy_thresh(img, details): #details는 하나의 sub-band.
    #주어진 sub-band에 대해서 threshold를 추정하는 함수
    delta_MAD = np.median(np.abs(details))/0.6745
    thresh = delta_MAD * np.sqrt(2*np.log(img.size))
    return thresh

def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.
    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.
    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma


def _wavelet_threshold(image, wavelet, ext_mode='symmetric', axes=(-2,-1), method=None, threshold=None,
                       sigma=None, thresh_mode='soft', wavelet_levels=None):
    """Perform wavelet thresholding.
    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_, "VisuShrink" [2]_, "Fuzzy" [3]_ and "new_Visu" [4]_. If it is set to None, a
        user-specified ``threshold`` must be supplied instead.
    threshold : float, optional
        The thresholding value to apply during wavelet coefficient
        thresholding. The default value (None) uses the selected ``method`` to
        estimate appropriate threshold(s) for noise removal.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).
    Returns
    -------
    out : ndarray
        Denoised image.
    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    .. [3] Mario Mastriani. "Fuzzy thresholding in wavelet domain 
           for speckle reduction in Synthetic Aperture Radar images"
           Cornell University https://arxiv.org/abs/1608.00277
    .. [4] Lu Jing-yi,1,2Lin Hong,2Ye Dong ,1and Zhang Yan-sheng1,2.
           "A New Wavelet Threshold Function and Denoising Application"
           Mathematical Problems in Engineering, Volume 2016 | Article ID 3195492 | 
           https://doi.org/10.1155/2016/3195492
    """
    
    original_extent = tuple(slice(s) for s in image.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)

        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels, mode=ext_mode, axes=(-2,-1))
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        warn(f'Thresholding method {method} selected. The '
             f'user-specified threshold will be ignored.')

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError(
                "If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(image, sigma)
        elif method == "Fuzzy":
            threshold = [{key: _fuzzy_thresh(image, level[key]) for key in level} for level in dcoeffs]
        elif method == 'new_Visu':
            threshold = [{key: _universal_thresh(image, sigma)/(np.log2(wavelet_levels+1-i)) for key in level} #key: 'ad' ,'da', 'dd'
                         for i,level in enumerate(dcoeffs)] #i:0~n-1
        else:
            raise ValueError(f'Unrecognized method: {method}')

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=thresh_mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=thresh_mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]