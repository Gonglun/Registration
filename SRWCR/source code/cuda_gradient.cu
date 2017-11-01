/*
   The main function calculates A-SRWCR and its gradient.
   The input:
        in1-----the fixed image;
		in2-----the moving image;
		Transform_Image----the transformed image;
		X-----the control points;
		gradientu-----the gradinet
		int_constant---{SIZEX,SIZEY,SIZEZ,nx,ny,nz};
        float_constant--{spacex,spacey,spacez};
*/
extern "C" double GPU_gradient_TPS(float *in1,float *in2,float *Transform_Image,const lbfgsfloatval_t *X,lbfgsfloatval_t *gradientu,int *int_constant,float *float_constant,lbfgs_client_parm *instance)
{
   float *dev_in1,*dev_in2,*dev_Transform_Image,*dev_float_constant,*dev_gradient_TI,*dev_gradient_image,*dev_transformxyz,*dev_m_f;
   double *dev_var_block,*dev_pr;
   uint *dev_pxyz;
   lbfgsfloatval_t *dev_X,*dev_gradient;
   int *dev_int_constant;
   int image_SIZE=int_constant[3]*int_constant[4]*int_constant[5];
   int grid_SIZE=int_constant[0]*int_constant[1]*int_constant[2];
   int bit_image=image_SIZE*sizeof(float);
   int bit_grid_d=grid_SIZE*sizeof(double);
   int bit_variable_d=grid_SIZE*3*sizeof(double);
  
   float *dev_BsplineX,*dev_BsplineY,*dev_BsplineZ;
   dev_in1=instance[0].dev_farray_1[1];
   dev_in2=instance[0].dev_farray_1[0];
   dev_float_constant=instance[0].dev_farray_1[2];
   dev_BsplineX=instance[0].dev_farray_1[3];     // the B-spline coefficients of all voxel in X direction
   dev_BsplineY=instance[0].dev_farray_1[4];    //  the B-spline coefficients of all voxel in Y direction
   dev_BsplineZ=instance[0].dev_farray_1[5];   //   the B-spline coefficients of all voxel in Z direction
   dev_pxyz=instance[0].dev_uint[1];
   HANDLE_ERROR(cudaMalloc((void **)&dev_X,bit_variable_d));//dev_X
   HANDLE_ERROR(cudaMalloc((void **)&dev_Transform_Image,bit_image));  //dev_Transform_Image
   HANDLE_ERROR(cudaMalloc((void **)&dev_int_constant,sizeof(int)*8));
   HANDLE_ERROR(cudaMalloc((void **)&dev_transformxyz,3*bit_image));

  
   //Transform the moving image--kernel 1
   int threads =512;
   int blocks=min(32,(image_SIZE+threads-1)/threads);
   HANDLE_ERROR(cudaMemcpy(dev_X,X,bit_variable_d,cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(dev_int_constant,int_constant,8*sizeof(int),cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(dev_float_constant,float_constant,4*sizeof(float),cudaMemcpyHostToDevice));
   cudaBspline_transform<<<blocks,threads>>>(dev_in1,dev_Transform_Image,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,dev_X,dev_transformxyz,dev_pxyz,instance[0].mode_Interpolation);
   cudaError_t err=cudaGetLastError();
   if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
   HANDLE_ERROR(cudaMemcpy(Transform_Image,dev_Transform_Image,bit_image,cudaMemcpyDeviceToHost));

  //calculate SRWCR
  HANDLE_ERROR(cudaMalloc((void **)&dev_var_block,bit_grid_d));//dev_var_block
  HANDLE_ERROR(cudaMalloc((void **)&dev_m_f,sizeof(float)*grid_SIZE*bin));//dev_m_f
  dev_pr=instance[0].dev_farray_1_d[0];  // this array contains the probability of each subregion
  //1£©calculate pab of each subregion
  double *dev_pab; //grid_SIZE*bin*bin
  int size_pab=grid_SIZE*bin*bin;
  double *pab=new double [size_pab];
  HANDLE_ERROR(cudaMalloc((void **)&dev_pab,sizeof(double)*size_pab));
  cudaMemset(dev_pab,0,sizeof(double)*size_pab);
  uint *dev_PerBlockCount=instance[0].dev_uint[0];  //this array contains the start and end indexes of each subregion
  double *dev_weight,*dev_hm;
  dev_hm=instance[0].dev_farray_1_d[1]; // the parzen-window coefficients of the fixed image  
  HANDLE_ERROR(cudaMalloc((void **)&dev_weight,2*sizeof(double)*image_SIZE));
  threads=1024;
  blocks=min(32,(image_SIZE+threads-1)/threads);
  cudaGPUHm<<<blocks,threads>>>(dev_Transform_Image,dev_weight,image_SIZE); // compute the parzen-window coefficients of the transformed image 
  err=cudaGetLastError();
  if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
  //1)--compute the regional joint PDFs. kernel 2
  threads=512;
  blocks=grid_SIZE;
  cudaPabCaculateBlock_Auto<<<blocks,threads>>>(dev_in2,dev_Transform_Image,dev_PerBlockCount,dev_weight,dev_hm,dev_pab,dev_pr,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ);
  err=cudaGetLastError();
  if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
  HANDLE_ERROR(cudaMemcpy(pab,dev_pab,sizeof(double)*size_pab,cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_weight));
  //2)---calculate SRWCR in CPU
  double *var_block=new double [grid_SIZE];
  memset(var_block,0,bit_grid_d);
  float *m_f=new float [grid_SIZE*bin];
  double weight[bin],hm[bin];
  double t,sum_square_M_block,wx,ave_block;
  double cr=0;
  double *pr=instance[0].pr;
  int bid_index;
  for(int bid=0;bid<grid_SIZE;bid++)
  {
	 bid_index=bid*bin*bin;
	 sum_square_M_block=0;
	 ave_block=0;
     for(int i=0;i<bin;i++)
     {
        t=0;
	    for(int j=0;j<bin;j++)
	       t+=pab[bid_index+j*bin+i];
	    weight[i]=t; //pb
	    ave_block+=i*weight[i];
		sum_square_M_block+=i*i*weight[i];
     }
     var_block[bid]=sum_square_M_block-ave_block*ave_block;
	 for(int i=0;i<bin;i++)
     {
        t=0;
	    for(int j=0;j<bin;j++)
	       t+=pab[bid_index+i*bin+j];
	    hm[i]=t;  //pa
      }
	  wx=0;
	  for(int i=0;i<bin;i++)
      {
         t=0;
	     for(int j=0;j<bin;j++)
            t+=j*pab[bid_index+i*bin+j];
	     if(hm[i]>0)
	        m_f[bid*bin+i]=t/hm[i];
	     else
		    m_f[bid*bin+i]=0;

		 wx+=hm[i]*m_f[bid*bin+i]*m_f[bid*bin+i];
        } 
	    cr+=pr[bid]*(sum_square_M_block-wx)/(var_block[bid]+INF);
   }
  HANDLE_ERROR(cudaMemcpy(dev_var_block,var_block,bit_grid_d,cudaMemcpyHostToDevice));   // these results will be used to compute the gradient
  HANDLE_ERROR(cudaMemcpy(dev_m_f,m_f,sizeof(float)*grid_SIZE*bin,cudaMemcpyHostToDevice)); // these results will be used to compute the gradient
  HANDLE_ERROR(cudaFree(dev_pab));
  free(pab);
  free(var_block);
  free(m_f);
  //calculate the gradient of moving image and the derivatives of SRWCR with respect to all intensities. kernel 3
  HANDLE_ERROR(cudaMalloc((void **)&dev_gradient,bit_variable_d));// the gradient of X 
  HANDLE_ERROR(cudaMalloc((void **)&dev_gradient_image,bit_image*3));//the gradient of transformed image along X,Y and Z direction.
  HANDLE_ERROR(cudaMalloc((void **)&dev_gradient_TI,bit_image));// the derivatives of SRWCR with respect to all intensities
  threads =512;
  blocks=min(32,(image_SIZE+threads-1)/threads);
  cudaImageGradient_TPS<<<blocks,threads>>>(dev_in1,dev_in2,dev_hm,dev_Transform_Image,dev_gradient_TI,dev_gradient_image,dev_var_block,dev_m_f,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,dev_transformxyz,instance[0].mode_Interpolation);
  HANDLE_ERROR(cudaFree(dev_m_f));
  HANDLE_ERROR(cudaFree(dev_var_block));
  HANDLE_ERROR(cudaFree(dev_transformxyz));
  HANDLE_ERROR(cudaFree(dev_Transform_Image));
  err=cudaGetLastError();
  if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
  //calculate the gradient of SRWCR---kernel 4
  int MAX_gridsize=65535;
  if(grid_SIZE<MAX_gridsize) // judge if the number of control points exceeds the max block size 
  {
    blocks=grid_SIZE;
    threads=512;
    cudaGradient_block_TPS<<<blocks,threads>>>(dev_gradient_TI,dev_gradient_image,dev_gradient,dev_PerBlockCount,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,0);
    err=cudaGetLastError();
    if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
  }
  else   //divide the control points 
  {
	blocks=(int)(grid_SIZE/3);
	threads=512;
    cudaGradient_block_TPS<<<blocks,threads>>>(dev_gradient_TI,dev_gradient_image,dev_gradient,dev_PerBlockCount,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,0);
    err=cudaGetLastError();
    if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
	blocks=(int)(grid_SIZE/3);
	threads=512;
    cudaGradient_block_TPS<<<blocks,threads>>>(dev_gradient_TI,dev_gradient_image,dev_gradient,dev_PerBlockCount,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,grid_SIZE/3);
	err=cudaGetLastError();
    if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
	blocks=(int)(grid_SIZE-(grid_SIZE/3)*2);
	threads=512;
    cudaGradient_block_TPS<<<blocks,threads>>>(dev_gradient_TI,dev_gradient_image,dev_gradient,dev_PerBlockCount,dev_int_constant,dev_float_constant,dev_BsplineX,dev_BsplineY,dev_BsplineZ,(grid_SIZE/3)*2);
    err=cudaGetLastError();
    if(err!=cudaSuccess)
	  printf("Error: %s\n",cudaGetErrorString(err));
  }
  HANDLE_ERROR(cudaMemcpy(gradientu,dev_gradient,bit_variable_d,cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_gradient_TI));
  HANDLE_ERROR(cudaFree(dev_gradient_image));
  
  HANDLE_ERROR(cudaFree(dev_gradient));
  HANDLE_ERROR(cudaFree(dev_X));
  HANDLE_ERROR(cudaFree(dev_int_constant));

  return cr;
}

__global__ void  cudaBspline_transform(float *in1,float *Transform_Image,int *int_constant,float *double_constant,float *BsplineX,float *BsplineY,float *BsplineZ,double *X,float *transformxyz,uint *pxyz,int mode)
{
   int tid=threadIdx.x;
   int bid=blockIdx.x;
   int K=bid*blockDim.x+tid;
   int h,h1,h2;
   float t,sx,sy,sz;
   int indx,indy,indz,px,py,pz;
   int temp_index;
   int image_SIZE=int_constant[3]*int_constant[4]*int_constant[5];
   int grid_SIZE=int_constant[0]*int_constant[1]*int_constant[2];
   int Isize[3]={int_constant[3],int_constant[4],int_constant[5]};
   int SIZEX=int_constant[3],SIZEY=int_constant[4],SIZEZ=int_constant[5];
   int black, cubic;
   if(mode==0||mode==2){ black = false; } else { black = true; }
   if(mode==0||mode==1){ cubic = false; } else { cubic = true; }
   while(K<image_SIZE)
   {
       h1=pxyz[K+2*image_SIZE];
	   h=pxyz[K+image_SIZE];
	   h2=pxyz[K];
	   t=h2*1.0/double_constant[0];px=(int)(t);
	   t=h*1.0/double_constant[1];py=(int)(t);
	   t=h1*1.0/double_constant[2];pz=(int)(t);
	   sx=0;sy=0;sz=0;
	   for(indx=0;indx<4;indx++)
	     for(indy=0;indy<4;indy++)
		   for(indz=0;indz<4;indz++)
		   {
              if((px+indx)>=0&&(px+indx)<=int_constant[0]&&(py+indy)>=0&&(py+indy)<=int_constant[1]&&(pz+indz)>=0&&(pz+indz)<=int_constant[2])
              {
				temp_index=(pz+indz)*int_constant[0]*int_constant[1]+int_constant[0]*(py+indy)+(px+indx);
			    sx+=1.0*BsplineX[K+indx*image_SIZE]*BsplineY[K+indy*image_SIZE]*BsplineZ[K+indz*image_SIZE]*X[temp_index];
			    sy+=1.0*BsplineX[K+indx*image_SIZE]*BsplineY[K+indy*image_SIZE]*BsplineZ[K+indz*image_SIZE]*X[grid_SIZE+temp_index];
			    sz+=1.0*BsplineX[K+indx*image_SIZE]*BsplineY[K+indy*image_SIZE]*BsplineZ[K+indz*image_SIZE]*X[2*grid_SIZE+temp_index];
			   }
			}
        Transform_Image[K]=interpolate_3d_float_gray_GPU(sx, sy, sz, Isize, in1,cubic,black);
		transformxyz[K]=sx;
		transformxyz[K+image_SIZE]=sy;
		transformxyz[K+2*image_SIZE]=sz;
		K+=blockDim.x*gridDim.x;
   }
}
__global__ void cudaPabCaculateBlock_Auto(float *in2,float *Transform_Image, uint *PerBlockCount,double *weight,double *hm,double *pab,double *pr,int *int_constant,float *float_constant,float *BsplineX,float *BsplineY,float *BsplineZ)
{
   int tid=threadIdx.x;
   int bid=blockIdx.x;
   int K=tid;
   int sx=PerBlockCount[bid*9+3]-PerBlockCount[bid*9]+1;
   int sy=PerBlockCount[bid*9+4]-PerBlockCount[bid*9+1]+1;
   int sz=PerBlockCount[bid*9+5]-PerBlockCount[bid*9+2]+1;
   int lx,ly,lz;
   int px,py,pz;
   int pixel_index;
   int pixel_count=sx*sy*sz;
   int image_SIZE=int_constant[3]*int_constant[4]*int_constant[5];
   int i=PerBlockCount[bid*9+6];
   int j=PerBlockCount[bid*9+7];
   int j1=PerBlockCount[bid*9+8];
   double t;
   double wx;
   while(K<pixel_count)
   {
      pz=(int)(K/(sx*sy));
	  lz=PerBlockCount[bid*9+2]+pz;
	  py=(int)(K-pz*sx*sy)/sx;
	  ly=PerBlockCount[bid*9+1]+py;
	  px=(int)(K-pz*sx*sy-py*sx);
	  lx=PerBlockCount[bid*9]+px;
	  pixel_index=lz*int_constant[3]*int_constant[4]+ly*int_constant[3]+lx;
	  t=lx*1.0/float_constant[0];px=(int)(t);
	  t=ly*1.0/float_constant[1];py=(int)(t);
	  t=lz*1.0/float_constant[2];pz=(int)(t);
	  wx=BsplineX[pixel_index+(i-px)*image_SIZE]*BsplineY[pixel_index+(j-py)*image_SIZE]*BsplineZ[pixel_index+(j1-pz)*image_SIZE];
	  px=(int)(Transform_Image[pixel_index]);
	  py=(int)(in2[pixel_index]);
	  t=weight[pixel_index]*hm[pixel_index]*wx/(image_SIZE*pr[bid]+INF);
	  atomicAdd(&(pab[bid*bin*bin+px*bin+py]),t); 
	  if(px<bin-1)
	  {
	     t=weight[pixel_index+image_SIZE]*hm[pixel_index]*wx/(image_SIZE*pr[bid]+INF);
	     atomicAdd(&(pab[bid*bin*bin+(px+1)*bin+py]),t);
	  }
	  if(py<bin-1)
	  {
	    t=weight[pixel_index]*hm[pixel_index+image_SIZE]*wx/(image_SIZE*pr[bid]+INF);
	    atomicAdd(&(pab[bid*bin*bin+px*bin+py+1]),t);
	  }
	  if(px<(bin-1)&&py<(bin-1))
	  {
	    t=weight[pixel_index+image_SIZE]*hm[pixel_index+image_SIZE]*wx/(image_SIZE*pr[bid]+INF);
	    atomicAdd(&(pab[bid*bin*bin+(px+1)*bin+py+1]),t);
	  }
	  K+=blockDim.x;
   }
}
__global__ void cudaImageGradient_TPS(float *in1,float *in2,double *hm,float *Transform_Image,float *gradient_TI,float *gradient_image,double *var_block,float *m_f,int *int_constant,float *float_constant,float *BsplineX,float *BsplineY,float *BsplineZ,float *transformxyz,int mode)
{
   int tid=threadIdx.x;
   int bid=blockIdx.x;
   int K=bid*blockDim.x+tid;
   int image_SIZE=int_constant[3]*int_constant[4]*int_constant[5];
   float weight[2];
   float sx,sy,sz;
   float step=0.005;
   float add,min;
   int black,cubic;
   int h1,h,h2;
   int px,py,pz;
   int offsetx,offsety,offsetz,offset_bid;
   double wx,t;
   double temp1,temp2,temp3;
   int Isize[3]={int_constant[3],int_constant[4],int_constant[5]};
   if(mode==0||mode==2){ black = false; } else { black = true; }
   if(mode==0||mode==1){ cubic = false; } else { cubic = true; }

   while(K<image_SIZE)
   {
	   sx=transformxyz[K];sy=transformxyz[K+image_SIZE];sz=transformxyz[K+2*image_SIZE];
	   // the gradient of moving image in X direction
	   add=interpolate_3d_float_gray_GPU(sx+step, sy, sz, Isize, in1,cubic,black);
       min=interpolate_3d_float_gray_GPU(sx-step, sy, sz, Isize, in1,cubic,black);
       gradient_image[K]=(add-min)/(2*step);
      //the gradient of moving image in Y direction
      add=interpolate_3d_float_gray_GPU(sx, sy+step, sz, Isize, in1,cubic,black);
      min=interpolate_3d_float_gray_GPU(sx, sy-step, sz, Isize, in1,cubic,black);
      gradient_image[K+image_SIZE]=(add-min)/(2*step);
      //the gradient of moving image in Z direction
     add=interpolate_3d_float_gray_GPU(sx, sy, sz+step, Isize, in1,cubic,black);
     min=interpolate_3d_float_gray_GPU(sx, sy, sz-step, Isize, in1,cubic,black);
     gradient_image[K+2*image_SIZE]=(add-min)/(2*step);

	  //the derivatives of SRWCR with respect to all intensities
	 h1=(int)(K/(int_constant[3]*int_constant[4]));
	 h=(int)((K-h1*int_constant[3]*int_constant[4])/int_constant[3]);
	 h2=(int)(K-h1*int_constant[3]*int_constant[4]-h*int_constant[3]);
	 t=h2*1.0/float_constant[0];px=(int)(t);
	 t=h*1.0/float_constant[1];py=(int)(t);
	 t=h1*1.0/float_constant[2];pz=(int)(t);
	 temp3=0;
	 h1=(int)(in2[K]);
	 h=(int)(Transform_Image[K]);
	 wx=h-Transform_Image[K];
	 if(wx>-1&&wx<=-0.5)
		 weight[0]=3.6*wx+3.7;
	 else if(wx>-0.5&&wx<0)
		 weight[0]=-3.6*wx+0.1;
	 else
		 weight[0]=0;
	 wx=h+1-Transform_Image[K];
	 if(wx>0&&wx<=0.5)
		 weight[1]=-3.6*wx-0.1;
	 else if(wx>0.5&&wx<1)
		 weight[1]=3.6*wx-3.7;
	 else
		 weight[1]=0;
	 for(int i=0;i<4;i++)
	   for(int j=0;j<4;j++)
		 for(int k=0;k<4;k++)
		 {
		    wx=BsplineX[K+i*image_SIZE]*BsplineY[K+j*image_SIZE]*BsplineZ[K+k*image_SIZE];
			offsetx=i+px;offsety=j+py;offsetz=k+pz;
			offset_bid=offsetz*(int_constant[0]*int_constant[1])+offsety*int_constant[0]+offsetx;

			temp1=0;
			temp1+=m_f[offset_bid*bin+h]*m_f[offset_bid*bin+h]*weight[0];
			if(h<bin-1)
			   temp1+=m_f[offset_bid*bin+h+1]*m_f[offset_bid*bin+h+1]*weight[1];
			temp2=0;
			temp2+=2*h1*m_f[offset_bid*bin+h]*weight[0]*hm[K];
			if(h<bin-1)
				temp2+=2*h1*m_f[offset_bid*bin+h+1]*weight[1]*hm[K];
			if(h1<bin-1)
				temp2+=2*(h1+1)*m_f[offset_bid*bin+h]*weight[0]*hm[K+image_SIZE];
			if(h<(bin-1)&&h1<(bin-1))
				temp2+=2*(h1+1)*m_f[offset_bid*bin+h+1]*weight[1]*hm[K+image_SIZE];

			temp3+=(temp1-temp2)*wx*(-1)/(image_SIZE*var_block[offset_bid]+INF);
		 }
     gradient_TI[K]=temp3;
	 K+=blockDim.x*gridDim.x;
   }
}
__global__ void cudaGradient_block_TPS(float *gradient_TI,float *gradient_image,double *gradientu,uint *PerBlockCount,int *int_constant,float *float_constant,float *BsplineX,float *BsplineY,float *BsplineZ,int start_bid)
{
   __shared__ double temp_gradientx[1024];
   __shared__ double temp_gradienty[1024];
   __shared__ double temp_gradientz[1024];
   int bid=blockIdx.x+start_bid;
   int tid=threadIdx.x;
   int K=tid;
   int sx=PerBlockCount[bid*9+3]-PerBlockCount[bid*9]+1;
   int sy=PerBlockCount[bid*9+4]-PerBlockCount[bid*9+1]+1;
   int sz=PerBlockCount[bid*9+5]-PerBlockCount[bid*9+2]+1;
   int lx,ly,lz;
   int px,py,pz;
   int pixel_index;
   int pixel_count=sx*sy*sz;
   int image_SIZE=int_constant[3]*int_constant[4]*int_constant[5];
   int grid_SIZE=int_constant[0]*int_constant[1]*int_constant[2];
   int i=PerBlockCount[bid*9+6];
   int j=PerBlockCount[bid*9+7];
   int j1=PerBlockCount[bid*9+8];
   double temp1=0,temp2=0,temp3=0;
   double same_sim2,t;
   double gT;
   while(K<pixel_count)
   {
      pz=(int)(K/(sx*sy));
	  lz=PerBlockCount[bid*9+2]+pz;
	  py=(int)(K-pz*sx*sy)/sx;
	  ly=PerBlockCount[bid*9+1]+py;
	  px=(int)(K-pz*sx*sy-py*sx);
	  lx=PerBlockCount[bid*9]+px;
	  pixel_index=lz*int_constant[3]*int_constant[4]+ly*int_constant[3]+lx;
	  t=lx*1.0/float_constant[0];px=(int)(t);
	  t=ly*1.0/float_constant[1];py=(int)(t);
	  t=lz*1.0/float_constant[2];pz=(int)(t);
	  gT=BsplineX[pixel_index+(i-px)*image_SIZE]*BsplineY[pixel_index+(j-py)*image_SIZE]*BsplineZ[pixel_index+(j1-pz)*image_SIZE];
	  same_sim2=gradient_TI[pixel_index]*gT; 
	  temp1+=same_sim2*gradient_image[pixel_index];
	  temp2+=same_sim2*gradient_image[pixel_index+image_SIZE];
	  temp3+=same_sim2*gradient_image[pixel_index+2*image_SIZE];

	  K+=blockDim.x;
   }
   temp_gradientx[tid]=temp1;
   temp_gradienty[tid]=temp2;
   temp_gradientz[tid]=temp3;
   __syncthreads();
   i=blockDim.x/2;
   while(i!=0)
   {
	 if(tid<i)
	 {
	    temp_gradientx[tid]+=temp_gradientx[tid+i];
		temp_gradienty[tid]+=temp_gradienty[tid+i];
		temp_gradientz[tid]+=temp_gradientz[tid+i];
	  }
      __syncthreads();
	  i/=2;
	}
	if(tid==0)
	{
	    gradientu[bid]=temp_gradientx[0];
		gradientu[bid+grid_SIZE]=temp_gradienty[0];
		gradientu[bid+2*grid_SIZE]=temp_gradientz[0];
	}
}