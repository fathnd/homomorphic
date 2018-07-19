#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialDilatedMaxPooling.c"
#else

static inline void THNN_(SpatialDilatedMaxPooling_shapeCheck)(
	THTensor *input, THTensor *gradOutput, THIndexTensor *indices,
	int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW,
	int64_t dilationH, int64_t dilationW, bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationH > 0 && dilationW > 0, 12,
             "dilation should be greater than zero, but got dilationH: %d dilationW: %d",
             dilationH, dilationW);

  int64_t ndim = input->dim();
  int64_t dimf = 0;
  int64_t dimh = 1;
  int64_t dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(!input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
		"non-empty 3D or 4D input tensor expected but got: %s");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
	     "pad should be smaller than half of kernel size, but got "
	     "padW = %d, padH = %d, kW = %d, kH = %d",
	     padW, padH, kW, kH);

  int64_t nInputPlane = input->size[dimh-1];
  int64_t inputHeight = input->size[dimh];
  int64_t inputWidth = input->size[dimw];
  int64_t outputHeight, outputWidth;
  int64_t nOutputPlane = nInputPlane;

  if (ceil_mode)
  {
    outputHeight = (int64_t)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int64_t)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputHeight = (int64_t)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int64_t)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). "
	    "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,inputHeight,inputWidth,nInputPlane,outputHeight,outputWidth);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
  if (indices != NULL) {
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimw, outputWidth);
  }
}

static void THNN_(SpatialDilatedMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *ind_p,
          int64_t nslices,
          int64_t iwidth,
          int64_t iheight,
          int64_t owidth,
          int64_t oheight,
          int64_t kW,
          int64_t kH,
          int64_t dW,
          int64_t dH,
          int64_t padW,
          int64_t padH,
          int64_t dilationW,
          int64_t dilationH
          )
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    int64_t i, j;
    real *ip = input_p   + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        int64_t hstart = i * dH - padH;
        int64_t wstart = j * dW - padW;
        int64_t hend = fminf(hstart + (kH - 1) * dilationH + 1, iheight);
        int64_t wend = fminf(wstart + (kW - 1) * dilationW + 1, iwidth);
        while(hstart < 0)
          hstart += dilationH;
        while(wstart < 0)
          wstart += dilationW;

        /* local pointers */
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        THIndex_t *indp = ind_p   + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        int64_t maxindex = -1;
        real maxval = -THInf;
        int64_t tcntr = 0;
        int64_t x,y;
        for(y = hstart; y < hend; y += dilationH)
        {
          for(x = wstart; x < wend; x += dilationW)
          {
            tcntr = y*iwidth + x;
            real val = *(ip + tcntr);
            if ((val > maxval) || std::isnan(val))
            {
              maxval = val;
              maxindex = tcntr;
            }
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max */
        *indp = maxindex + TH_INDEX_BASE;
      }
    }
  }
}

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int64_t kW,
          int64_t kH,
          int64_t dW,
          int64_t dH,
          int64_t padW,
          int64_t padH,
          int64_t dilationW,
          int64_t dilationH,
          bool ceil_mode)
{

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t nbatch = 1;
  int64_t nInputPlane;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t outputHeight;
  int64_t outputWidth;
  real *input_data;
  real *output_data;
  THIndex_t *indices_data;

  THNN_(SpatialDilatedMaxPooling_shapeCheck)
    (input, NULL, NULL, kH, kW, dH, dW,
     padH, padW, dilationH, dilationW, ceil_mode);

  if (input->dim() == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nInputPlane = input->size[dimh-1];
  inputHeight = input->size[dimh];
  inputWidth = input->size[dimw];
  if (ceil_mode)
  {
    outputHeight = (int64_t)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int64_t)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputHeight = (int64_t)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int64_t)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->dim() == 3)
  {
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize3d)(indices,  nInputPlane, outputHeight, outputWidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    THNN_(SpatialDilatedMaxPooling_updateOutput_frame)
      (input_data, output_data,
       indices_data,
       nInputPlane,
       inputWidth, inputHeight,
       outputWidth, outputHeight,
       kW, kH, dW, dH,
       padW, padH,
       dilationW, dilationH
       );
  }
  else
  {
    int64_t p;

    THTensor_(resize4d)(output, nbatch, nInputPlane, outputHeight, outputWidth);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize4d)(indices, nbatch, nInputPlane, outputHeight, outputWidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialDilatedMaxPooling_updateOutput_frame)
	(input_data+p*nInputPlane*inputWidth*inputHeight,
	 output_data+p*nInputPlane*outputWidth*outputHeight,
	 indices_data+p*nInputPlane*outputWidth*outputHeight,
	 nInputPlane,
	 inputWidth, inputHeight,
	 outputWidth, outputHeight,
	 kW, kH, dW, dH,
	 padW, padH,
	 dilationW, dilationH
	 );
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          THIndex_t *ind_p,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int64_t dW,
          int64_t dH)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nInputPlane; k++)
  {
    real *gradInput_p_k = gradInput_p + k*inputWidth*inputHeight;
    real *gradOutput_p_k = gradOutput_p + k*outputWidth*outputHeight;
    THIndex_t *ind_p_k = ind_p + k*outputWidth*outputHeight;

    /* calculate max points */
    int64_t i, j;
    for(i = 0; i < outputHeight; i++)
    {
      for(j = 0; j < outputWidth; j++)
      {
        /* retrieve position of max */
        int64_t maxp = ind_p_k[i*outputWidth + j] - TH_INDEX_BASE;
	if (maxp != -1) {
	  /* update gradient */
	  gradInput_p_k[maxp] += gradOutput_p_k[i*outputWidth + j];
	}
      }
    }
  }
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int64_t kW,
          int64_t kH,
          int64_t dW,
          int64_t dH,
          int64_t padW,
          int64_t padH,
          int64_t dilationW,
          int64_t dilationH,
          bool ceil_mode)
{
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t nbatch = 1;
  int64_t nInputPlane;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t outputHeight;
  int64_t outputWidth;
  real *gradInput_data;
  real *gradOutput_data;
  THIndex_t *indices_data;

  THNN_(SpatialDilatedMaxPooling_shapeCheck)
    (input, gradOutput, indices, kH, kW, dH, dW,
     padH, padW, dilationH, dilationW, ceil_mode);

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->dim() == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nInputPlane = input->size[dimh-1];
  inputHeight = input->size[dimh];
  inputWidth = input->size[dimw];
  outputHeight = gradOutput->size[dimh];
  outputWidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->dim() == 3)
  {
    THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)
      (gradInput_data, gradOutput_data,
       indices_data,
       nInputPlane,
       inputWidth, inputHeight,
       outputWidth, outputHeight,
       dW, dH);
  }
  else
  {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)
	(gradInput_data+p*nInputPlane*inputWidth*inputHeight,
	 gradOutput_data+p*nInputPlane*outputWidth*outputHeight,
	 indices_data+p*nInputPlane*outputWidth*outputHeight,
	 nInputPlane,
	 inputWidth, inputHeight,
	 outputWidth, outputHeight,
	 dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
