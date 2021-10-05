import SimpleITK as sitk
import os

from math import pi


def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Scales: ", method.GetOptimizerScales())
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                          method.GetMetricValue(),
                                          method.GetOptimizerPosition()))

def exhaustive_initial(fixed_img, moving_img):
    """
    Performs the initial good guess of rotation in brute force manner.
    """

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    sample_per_axis = 12

    tx = sitk.Euler3DTransform()
    R.SetOptimizerAsExhaustive([sample_per_axis // 2, sample_per_axis // 2,
                                sample_per_axis // 4, 0, 0, 0])

    R.SetOptimizerScales(
        [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis,
         2.0 * pi / sample_per_axis, 1.0, 1.0, 1.0])

    # Initialize the transform with a translation and the center of
    # rotation from the moments of intensity.
    tx = sitk.CenteredTransformInitializer(fixed_img, moving_img, tx)
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed_img, moving_img)


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving_img)

    return out

def coregister_mutual_information(fixed_img_path, moving_img_path, numberOfBins=24, samplingPercentage=0.10):
    """
    Coregister the two 3-D images using mutual information 
    """

    # read images from given paths
    fixed_img = sitk.ReadImage(fixed_img_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)

    moving_img_init = exhaustive_initial(fixed_img, moving_img)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_img.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed_img, moving_img_init)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving_img_init)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    sitk.Show(cimg, "ImageRegistration4 Composition")

    return out