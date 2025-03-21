%LES parameters
pex = 0.25;
pey = 0.5;
pez = 1;
Lx  = 2*pi/pex;
Ly  = 2*pi/pey;
Lz  = 2*pi/pez;
Nx  = 864;
Ny  = 576;
Nz  = 144;
x   = linspace(0,Lx,Nx);
y   = linspace(0,Ly,Ny);
kx  = [0:Nx/2, -Nx/2+1:-1] * (2 * pi / Lx);
ky  = [0:Ny/2, -Ny/2+1:-1] * (2 * pi / Ly);
kx_3D=zeros(Nx,Ny,Nz);
for j=1:Ny
    for k=1:Nz
        kx_3D(:,j,k)=kx;
    end
end
ky_3D=zeros(Nx,Ny,Nz);
for i=1:Nx
    for k=1:Nz
        ky_3D(i,:,k)=ky;
    end
end

c_phase = 2;
k_wavno = 4;
Hbar    = 1;
Re      = 1000;
nu      = 1/Re;
%Initialize phase average quantities
  u_phase_avg    = zeros(Nx, Nz);
u_w_phase_avg    = zeros(Nx, Nz);
  v_phase_avg    = zeros(Nx, Nz);
v_w_phase_avg    = zeros(Nx, Nz);

  w_phase_avg    = zeros(Nx, Nz);
  p_phase_avg    = zeros(Nx, Nz);
p_w_phase_avg    = zeros(Nx, Nz);
  W_phase_avg    = zeros(Nx, Nz);
  U_phase_avg    = zeros(Nx, Nz);

 uW_phase_avg    = zeros(Nx, Nz);
 uU_phase_avg    = zeros(Nx, Nz);
 wU_phase_avg    = zeros(Nx, Nz);
 wW_phase_avg    = zeros(Nx, Nz);
 vU_phase_avg    = zeros(Nx, Nz);
 vW_phase_avg    = zeros(Nx, Nz);
 


eta_phase_avg    = zeros(Nx, 1 ); 
taup_13_phase_avg= zeros(Nx, Nz);
taup_11_phase_avg= zeros(Nx, Nz);

  tau13_phase_avg= zeros(Nx, Nz);
  tau31_phase_avg= zeros(Nx, Nz);
  tau11_phase_avg= zeros(Nx, Nz);
  tau33_phase_avg= zeros(Nx, Nz);

tau11_SGS_phase_avg= zeros(Nx, Nz);
tau22_SGS_phase_avg= zeros(Nx, Nz);
tau33_SGS_phase_avg= zeros(Nx, Nz);
tau12_SGS_phase_avg= zeros(Nx, Nz);
tau21_SGS_phase_avg= zeros(Nx, Nz);
tau13_SGS_phase_avg= zeros(Nx, Nz);
tau31_SGS_phase_avg= zeros(Nx, Nz);
tau32_SGS_phase_avg= zeros(Nx, Nz);
tau23_SGS_phase_avg= zeros(Nx, Nz);

tau11_nu_phase_avg= zeros(Nx, Nz);
tau22_nu_phase_avg= zeros(Nx, Nz);
tau33_nu_phase_avg= zeros(Nx, Nz);
tau12_nu_phase_avg= zeros(Nx, Nz);
tau21_nu_phase_avg= zeros(Nx, Nz);
tau13_nu_phase_avg= zeros(Nx, Nz);
tau31_nu_phase_avg= zeros(Nx, Nz);
tau32_nu_phase_avg= zeros(Nx, Nz);
tau23_nu_phase_avg= zeros(Nx, Nz);

tau11_nu_J_phase_avg= zeros(Nx, Nz);
tau22_nu_J_phase_avg= zeros(Nx, Nz);
tau33_nu_J_phase_avg= zeros(Nx, Nz);
tau12_nu_J_phase_avg= zeros(Nx, Nz);
tau21_nu_J_phase_avg= zeros(Nx, Nz);
tau13_nu_J_phase_avg= zeros(Nx, Nz);
tau31_nu_J_phase_avg= zeros(Nx, Nz);
tau32_nu_J_phase_avg= zeros(Nx, Nz);
tau23_nu_J_phase_avg= zeros(Nx, Nz);


 tau13_wave_phase_avg= zeros(Nx, Nz);
 tau31_wave_phase_avg= zeros(Nx, Nz);
 tau11_wave_phase_avg= zeros(Nx, Nz);
 tau33_wave_phase_avg= zeros(Nx, Nz);

 %TKE budget terms
 TKE=zeros(Nx,Nz);
 uu_w_phase_avg  = zeros(Nx, Nz);
 vv_w_phase_avg  = zeros(Nx, Nz);
 ww_phase_avg  = zeros(Nx, Nz);


epsilon_phase_avg = zeros(Nx, Nz);
epsilon_J_phase_avg = zeros(Nx, Nz);

epsilon_SGS_phase_avg = zeros(Nx, Nz);

S11_phase_avg = zeros(Nx, Nz);
S22_phase_avg = zeros(Nx, Nz);
S33_phase_avg = zeros(Nx, Nz);
S12_phase_avg = zeros(Nx, Nz);
S13_phase_avg = zeros(Nx, Nz);
S23_phase_avg = zeros(Nx, Nz);

%TKE viscous transport
upi_taup_nu_i1_TKE  = zeros(Nx,Nz);
upi_taup_SGS_i1_TKE = zeros(Nx,Nz);
upi_taup_nu_i3_TKE  = zeros(Nx,Nz);
upi_taup_SGS_i3_TKE = zeros(Nx,Nz);

u_tau11_nu_phase_avg=zeros(Nx,Nz);
v_tau21_nu_phase_avg=zeros(Nx,Nz);
w_tau31_nu_phase_avg=zeros(Nx,Nz);

u_tau13_nu_phase_avg=zeros(Nx,Nz);
v_tau23_nu_phase_avg=zeros(Nx,Nz);
w_tau33_nu_phase_avg=zeros(Nx,Nz);

%TKE SGS transport
u_tau11_SGS_phase_avg=zeros(Nx,Nz);
v_tau21_SGS_phase_avg=zeros(Nx,Nz);
w_tau31_SGS_phase_avg=zeros(Nx,Nz);

u_tau13_SGS_phase_avg=zeros(Nx,Nz);
v_tau23_SGS_phase_avg=zeros(Nx,Nz);
w_tau33_SGS_phase_avg=zeros(Nx,Nz);

%pressure transport
pp_Up_TKE=zeros(Nx,Nz);
pp_Wp_TKE=zeros(Nx,Nz);

pU_phase_avg=zeros(Nx,Nz);
pW_phase_avg=zeros(Nx,Nz);

%Convective transport
uiuiU_TKE=zeros(Nx,Nz);
uiuiW_TKE=zeros(Nx,Nz);

uiuiU_phase_avg=zeros(Nx,Nz);
uiuiW_phase_avg=zeros(Nx,Nz);



n_array = 3000000000:20000000:15980000000;

Nt = numel(n_array)

%tau13_tot = tau13_visc + taup_13 + tau13_wave + tau13_SGS + tau13_turb;

counter = 1;
for n = n_array
   tic
   counter

   %filename = sprintf('./Sol%014d.h5', n);
   filename = sprintf('/scratch.global/ren00115/fine_WOW_LES_1K/02_c2/Sol%014d.h5', n);
   time = h5read(filename,'/time' );
   
   eta  = h5read(filename,'/eta'  );
   zw   = h5read(filename,'/zw'   );
   zz   = h5read(filename,'/zz'   );
   dzw  = h5read(filename,'/dzw'  );
   dz   = h5read(filename,'/dz'   );

   dzw_3D=repmat(dzw',[Nx*Ny,1]);
   dzw_3D=reshape(dzw_3D,[Nx,Ny,Nz]);

   dz_3D=repmat(dz',[Nx*Ny,1]);
   dz_3D=reshape(dz_3D,[Nx,Ny,Nz]);

   pp   = h5read(filename,'/pp'  );
   u    = h5read(filename,'/u'   );
   v    = h5read(filename,'/v'   );
   w    = h5read(filename,'/w'   );
   nu_T_sgs    = h5read(filename,'/nu_t'   );
   
   z_phys = zeros(Nx,Nz);
   x_phys = zeros(Nx,Nz);
   for i=1:Nx
        z_phys(i,:) = eta(i,1)+(zw(:)/Hbar).*(Hbar-eta(i,1));%zw(:).*(Hbar-eta_phase_avg(i))+eta_phase_avg(i);
        x_phys(i,:) = x(i);
   end

   %zeta_x, zeta_z - Hao & Shen (2022)
   zetax_w = zeros(Nx,Ny,Nz);
   zetay_w = zeros(Nx,Ny,Nz);
   zetaz_w = zeros(Nx,Ny,Nz);
  
   eta_x = zeros(Nx,Ny);
   for j=1:Ny
     eta_hat_1       = fft(eta(1:Nx,j))/(Nx);
     eta_x(1:Nx,j)   = real(ifft(sqrt(-1)*kx'.*eta_hat_1))*(Nx);
   end
  
   eta_y = zeros(Nx,Ny);
   for i=1:Nx
     eta_hat_2         = fft(eta(i,1:Ny))/(Ny);
     eta_y(i,1:Ny)   = real(ifft(sqrt(-1)*ky.*eta_hat_2))*(Ny);
   end
   eta_y(:,Ny) = eta_y(:,1);

   for k=1:Nz
     zetax_w(:,:,k) = eta_x(:,:).*(zw(k)-Hbar)./(Hbar-eta(:,:)) ;%( eta_x(:,:) - zw(k) * eta_x(:,:) )./(eta(:,:) + Hbar);
     zetay_w(:,:,k) = eta_y(:,:).*(zw(k)-Hbar)./(Hbar-eta(:,:)) ;%( eta_y(:,:) - zw(k) * eta_y(:,:) )./(eta(:,:) + Hbar);
     zetaz_w(:,:,k) = Hbar./(Hbar-eta(:,:));%1./(eta(:,:) + Hbar);
   end
 
   %interpolate u & p to w-grid
   u_w  = zeros(Nx,Ny,Nz);
   v_w  = zeros(Nx,Ny,Nz);
   p_w  = zeros(Nx,Ny,Nz);
   nu_T_sgs_w  = zeros(Nx,Ny,Nz);
    
   u_w(:,:,  1   )  = u(:,:,1);
   u_w(:,:, Nz-1 )  = u(:,:,Nz);
   u_w(:,:, Nz )    = u(:,:,Nz);
   u_w(:,:,2:Nz-2)  = 0.5*(u(:,:,2:Nz-2) + u(:,:,3:Nz-1));

   v_w(:,:,  1   )  = v(:,:,1);
   v_w(:,:, Nz-1 )  = v(:,:,Nz);
   v_w(:,:, Nz   )  = v(:,:,Nz);
   v_w(:,:,2:Nz-2)  = 0.5*(v(:,:,2:Nz-2) + v(:,:,3:Nz-1));


   p_w(:,:,  1   )  = pp(:,:,1);
   p_w(:,:, Nz-1 )  = pp(:,:,Nz);
   p_w(:,:, Nz   )  = pp(:,:,Nz);
   p_w(:,:,2:Nz-2)  = 0.5*(pp(:,:,2:Nz-2) + pp(:,:,3:Nz-1));
   
   nu_T_sgs_w(:,:,  1   )  = nu_T_sgs(:,:,1);
   nu_T_sgs_w(:,:, Nz-1 )  = nu_T_sgs(:,:,Nz);
   nu_T_sgs_w(:,:, Nz   )  = nu_T_sgs(:,:,Nz);
   nu_T_sgs_w(:,:,2:Nz-2)  = 0.5*(nu_T_sgs(:,:,2:Nz-2) + nu_T_sgs(:,:,3:Nz-1));
   
   %Calc Sij
   Sij=zeros(Nx,Ny,Nz,9);
   %Sij=calc_Sij(x,y,z_phys,u_w,v_w,w,Lx,Ly,Nx,Ny,Nz);
   Sij=calc_Sij_3D(kx_3D,ky_3D,dzw_3D,zetaz_w,u_w,v_w,w,Nx,Ny,Nz,zw);
  
   %tau_ij_SGS, tau_ij_nu
   tau11_SGS=zeros(Nx,Ny,Nz);
   tau22_SGS=zeros(Nx,Ny,Nz);
   tau33_SGS=zeros(Nx,Ny,Nz);
   tau12_SGS=zeros(Nx,Ny,Nz);
   tau21_SGS=zeros(Nx,Ny,Nz);
   tau13_SGS=zeros(Nx,Ny,Nz);
   tau31_SGS=zeros(Nx,Ny,Nz);
   tau23_SGS=zeros(Nx,Ny,Nz);
   tau32_SGS=zeros(Nx,Ny,Nz);
   

   tau11_nu=zeros(Nx,Ny,Nz);
   tau22_nu=zeros(Nx,Ny,Nz);
   tau33_nu=zeros(Nx,Ny,Nz);
   tau12_nu=zeros(Nx,Ny,Nz);
   tau21_nu=zeros(Nx,Ny,Nz);
   tau13_nu=zeros(Nx,Ny,Nz);
   tau31_nu=zeros(Nx,Ny,Nz);
   tau23_nu=zeros(Nx,Ny,Nz);
   tau32_nu=zeros(Nx,Ny,Nz);

   tau11_nu_J=zeros(Nx,Ny,Nz);
   tau22_nu_J=zeros(Nx,Ny,Nz);
   tau33_nu_J=zeros(Nx,Ny,Nz);
   tau12_nu_J=zeros(Nx,Ny,Nz);
   tau21_nu_J=zeros(Nx,Ny,Nz);
   tau13_nu_J=zeros(Nx,Ny,Nz);
   tau31_nu_J=zeros(Nx,Ny,Nz);
   tau23_nu_J=zeros(Nx,Ny,Nz);
   tau32_nu_J=zeros(Nx,Ny,Nz);

   S11=zeros(Nx,Ny,Nz);
   S22=zeros(Nx,Ny,Nz);
   S33=zeros(Nx,Ny,Nz);
   S12=zeros(Nx,Ny,Nz);
   S13=zeros(Nx,Ny,Nz);
   S23=zeros(Nx,Ny,Nz);
   
   S11(:,:,:) = Sij(:,:,:,1);
   S22(:,:,:) = Sij(:,:,:,5);
   S33(:,:,:) = Sij(:,:,:,9);
   S12(:,:,:) = Sij(:,:,:,2);
   S13(:,:,:) = Sij(:,:,:,3);
   S23(:,:,:) = Sij(:,:,:,6);

   tau11_SGS(:,:,:)=-2*nu_T_sgs_w.*S11./zetaz_w;
   tau22_SGS(:,:,:)=-2*nu_T_sgs_w.*S22./zetaz_w;
   tau33_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*(S13.*zetax_w+S23.*zetay_w+S33.*zetaz_w);
   tau12_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*S12;
   tau21_SGS(:,:,:)=tau12_SGS(:,:,:);
   tau13_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*(S11.*zetax_w+S12.*zetay_w+S13.*zetaz_w);
   tau31_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*S13;
   tau23_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*(S12.*zetax_w+S22.*zetay_w+S23.*zetaz_w);
   tau32_SGS(:,:,:)=-2*nu_T_sgs_w./zetaz_w.*S23;

   tau11_nu(:,:,:)=-2*nu.*S11./zetaz_w;
   tau22_nu(:,:,:)=-2*nu.*S22./zetaz_w;
   tau33_nu(:,:,:)=-2*nu./zetaz_w.*(S13.*zetax_w+S23.*zetay_w+S33.*zetaz_w);
   tau12_nu(:,:,:)=-2*nu./zetaz_w.*S12;
   tau21_nu(:,:,:)=tau12_nu(:,:,:);
   tau13_nu(:,:,:)=-2*nu./zetaz_w.*(S11.*zetax_w+S12.*zetay_w+S13.*zetaz_w);
   tau31_nu(:,:,:)=-2*nu./zetaz_w.*S13;
   tau23_nu(:,:,:)=-2*nu./zetaz_w.*(S12.*zetax_w+S22.*zetay_w+S23.*zetaz_w);
   tau32_nu(:,:,:)=-2*nu./zetaz_w.*S23;


   tau11_nu_J(:,:,:)=2*nu./zetaz_w.*S11;
   tau22_nu_J(:,:,:)=2*nu./zetaz_w.*S22;
   tau33_nu_J(:,:,:)=2*nu./zetaz_w.*S33;
   tau12_nu_J(:,:,:)=2*nu./zetaz_w.*S12;
   tau21_nu_J(:,:,:)=2*nu./zetaz_w.*S12;
   tau13_nu_J(:,:,:)=2*nu./zetaz_w.*S13;
   tau31_nu_J(:,:,:)=2*nu./zetaz_w.*S13;
   tau23_nu_J(:,:,:)=2*nu./zetaz_w.*S23;
   tau32_nu_J(:,:,:)=2*nu./zetaz_w.*S23;
   
   
   % tau33_SGS(:,:,:)=-2*nu_T_sgs_w.*Sij(:,:,:,7).*zetax_w(:,:,:)./zetaz_w(:,:,:) ...
   %                  -2*nu_T_sgs_w.*Sij(:,:,:,8).*zetax_w(:,:,:)./zetaz_w(:,:,:) ...
   %                  -2*nu_T_sgs_w.*Sij(:,:,:,9);
   % 
   % tau13_SGS(:,:,:)=-2*nu_T_sgs_w.*Sij(:,:,:,1).*zetax_w(:,:,:)./zetaz_w(:,:,:) ...
   %                  -2*nu_T_sgs_w.*Sij(:,:,:,2).*zetay_w(:,:,:)./zetaz_w(:,:,:) ...
   %                  -2*nu_T_sgs_w.*Sij(:,:,:,3);
   % 
   % tau31_SGS(:,:,:)=-2*nu_T_sgs_w.*Sij(:,:,:,7)./zetaz_w(:,:,:);
   
   
   
   


   % tau13_nu(:,:,:)=-2*nu*Sij(:,:,:,1).*zetax_w(:,:,:)./zetaz_w(:,:,:) ...
   %                 -2*nu*Sij(:,:,:,2).*zetay_w(:,:,:)./zetaz_w(:,:,:) ...
   %                 -2*nu*Sij(:,:,:,3);
   % 
   % tau31_nu(:,:,:)=-2*nu*Sij(:,:,:,7)./zetaz_w(:,:,:);
   % 
   % tau11_nu(:,:,:)=-2*nu*Sij(:,:,:,1)./zetaz_w(:,:,:);
   % 
   % tau33_nu(:,:,:)=-2*nu*Sij(:,:,:,7).*zetax_w(:,:,:)./zetaz_w(:,:,:) ...
   %                 -2*nu*Sij(:,:,:,8).*zetay_w(:,:,:)./zetaz_w(:,:,:) ...
   %                 -2*nu*Sij(:,:,:,9);
   
   
   
   %Calculate contravariant velocity U_i
   W = zeros(Nx,Ny,Nz);
   U = zeros(Nx,Ny,Nz);
    
   %Calculate taup_ij       
   taup_13 = zeros(Nx,Ny,Nz);
   taup_11 = zeros(Nx,Ny,Nz);
   %taup_31 = 0;
   %taup_33 = same as pressure p;
   
   U(:,:,:) =   (u_w(:,:,:) - c_phase)./zetaz_w(:,:,:);
   W(:,:,:) =   (u_w(:,:,:) - c_phase).*(zetax_w(:,:,:)./zetaz_w(:,:,:)) ...
              + v_w(:,:,:).*(zetay_w(:,:,:)./zetaz_w(:,:,:)) ...
              + w(:,:,:);

   taup_13(:,:,:) = + p_w(:,:,:).*zetax_w(:,:,:)./zetaz_w(:,:,:);
   taup_11(:,:,:) = + p_w(:,:,:)./zetaz_w(:,:,:);
   
   %Viscous & SGS dissipation term
   epsilon=zeros(Nx,Ny,Nz);
   epsilon_J=zeros(Nx,Ny,Nz);

   epsilon_SGS=zeros(Nx,Ny,Nz);
  
   
   epsilon(:,:,:) = 2*nu*(S11.*S11+S22.*S22+S33.*S33+ ...
                       2*(S12.*S12+S13.*S13+S23.*S23) );
   epsilon_J(:,:,:) = 2*nu*(S11.*S11+S22.*S22+S33.*S33+ ...
                         2*(S12.*S12+S13.*S13+S23.*S23) )./zetaz_w;
   
   epsilon_SGS(:,:,:) = tau11_SGS.*S11+tau22_SGS.*S22+tau33_SGS.*S33 ...
                      + tau12_SGS.*S12+tau13_SGS.*S13+tau23_SGS.*S23 ...
                      + tau21_SGS.*S12+tau31_SGS.*S13+tau32_SGS.*S23;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   %average in y-direction

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   %u_i
   u_xz   = zeros(Nx,Nz);
   u_w_xz = zeros(Nx,Nz);
   v_xz   = zeros(Nx,Nz);
   v_w_xz = zeros(Nx,Nz);
   
   w_xz   = zeros(Nx,Nz);
   p_xz   = zeros(Nx,Nz);
   p_w_xz = zeros(Nx,Nz);  

   %U_i
   W_xz   = zeros(Nx,Nz);
   U_xz   = zeros(Nx,Nz);
   
   %uiUj
   uW_xz   = zeros(Nx,Nz);
   uU_xz   = zeros(Nx,Nz);
   wU_xz   = zeros(Nx,Nz);
   wW_xz   = zeros(Nx,Nz);
   vU_xz   = zeros(Nx,Nz);
   vW_xz   = zeros(Nx,Nz);
   
   %uiuj
   uu_w_xz   = zeros(Nx,Nz);
   vv_w_xz   = zeros(Nx,Nz);
   ww_xz   = zeros(Nx,Nz);
   
   %taup_ij
   taup_13_xz = zeros(Nx,Nz);
   taup_11_xz = zeros(Nx,Nz);

   %tau_SGS_ij
    tau11_SGS_xz= zeros(Nx, Nz);
    tau22_SGS_xz= zeros(Nx, Nz);
    tau33_SGS_xz= zeros(Nx, Nz);
    tau12_SGS_xz= zeros(Nx, Nz);
    tau21_SGS_xz= zeros(Nx, Nz);
    tau13_SGS_xz= zeros(Nx, Nz);
    tau31_SGS_xz= zeros(Nx, Nz);
    tau32_SGS_xz= zeros(Nx, Nz);
    tau23_SGS_xz= zeros(Nx, Nz);
    %tau_nu_ij
    tau11_nu_xz= zeros(Nx, Nz);
    tau22_nu_xz= zeros(Nx, Nz);
    tau33_nu_xz= zeros(Nx, Nz);
    tau12_nu_xz= zeros(Nx, Nz);
    tau21_nu_xz= zeros(Nx, Nz);
    tau13_nu_xz= zeros(Nx, Nz);
    tau31_nu_xz= zeros(Nx, Nz);
    tau32_nu_xz= zeros(Nx, Nz);
    tau23_nu_xz= zeros(Nx, Nz);

    tau11_nu_J_xz= zeros(Nx, Nz);
    tau22_nu_J_xz= zeros(Nx, Nz);
    tau33_nu_J_xz= zeros(Nx, Nz);
    tau12_nu_J_xz= zeros(Nx, Nz);
    tau21_nu_J_xz= zeros(Nx, Nz);
    tau13_nu_J_xz= zeros(Nx, Nz);
    tau31_nu_J_xz= zeros(Nx, Nz);
    tau32_nu_J_xz= zeros(Nx, Nz);
    tau23_nu_J_xz= zeros(Nx, Nz);


    %epsilon, Sij
    epsilon_xz = zeros(Nx,Nz);
    epsilon_J_xz = zeros(Nx,Nz);
    epsilon_SGS_xz = zeros(Nx,Nz);
    
    S11_xz = zeros(Nx,Nz);
    S22_xz = zeros(Nx,Nz);
    S33_xz = zeros(Nx,Nz);
    S12_xz = zeros(Nx,Nz);
    S13_xz = zeros(Nx,Nz);
    S23_xz = zeros(Nx,Nz);

    %TKE viscous & SGS transport
    u_tau11_nu_xz= zeros(Nx,Nz);
    v_tau21_nu_xz= zeros(Nx,Nz);
    w_tau31_nu_xz= zeros(Nx,Nz);

    u_tau13_nu_xz= zeros(Nx,Nz);
    v_tau23_nu_xz= zeros(Nx,Nz);
    w_tau33_nu_xz= zeros(Nx,Nz);

    u_tau11_SGS_xz= zeros(Nx,Nz);
    v_tau21_SGS_xz= zeros(Nx,Nz);
    w_tau31_SGS_xz= zeros(Nx,Nz);

    u_tau13_SGS_xz= zeros(Nx,Nz);
    v_tau23_SGS_xz= zeros(Nx,Nz);
    w_tau33_SGS_xz= zeros(Nx,Nz);
    
    %pressure transport
    pU_xz = zeros(Nx,Nz);
    pW_xz = zeros(Nx,Nz);
  
    %Convective transport term
    uiuiU_xz = zeros(Nx,Nz);
    uiuiW_xz = zeros(Nx,Nz);

    u_xz(:,:)   = squeeze(mean(  u(:,:,:),2));
    u_w_xz(:,:) = squeeze(mean(u_w(:,:,:),2));
    v_xz(:,:)   = squeeze(mean(  v(:,:,:),2));
    v_w_xz(:,:) = squeeze(mean(v_w(:,:,:),2));

   uu_w_xz(:,:) = squeeze(mean(u_w(:,:,:).*u_w(:,:,:),2));%TKE
   vv_w_xz(:,:) = squeeze(mean(v_w(:,:,:).*v_w(:,:,:),2));%TKE
     ww_xz(:,:) = squeeze(mean(  w(:,:,:).*w(:,:,:),2));%TKE
   
   w_xz(:,:)   = squeeze(mean(  w(:,:,:),2));

   p_xz(:,:)   = squeeze(mean( pp(:,:,:),2));
   p_w_xz(:,:) = squeeze(mean(p_w(:,:,:),2));  
   
   W_xz(:,:)   = squeeze(mean(W(:,:,:),2));
   U_xz(:,:)   = squeeze(mean(U(:,:,:),2)); 
   taup_13_xz(:,:) = squeeze(mean(taup_13(:,:,:),2));
   taup_11_xz(:,:) = squeeze(mean(taup_11(:,:,:),2));

        uW_xz(:,:) = squeeze(mean(u_w(:,:,:).*W(:,:,:),2));
        uU_xz(:,:) = squeeze(mean(u_w(:,:,:).*U(:,:,:),2));
        wW_xz(:,:) = squeeze(mean(w(:,:,:).*W(:,:,:),2));
        wU_xz(:,:) = squeeze(mean(w(:,:,:).*U(:,:,:),2));
        vW_xz(:,:) = squeeze(mean(v_w(:,:,:).*W(:,:,:),2));
        vU_xz(:,:) = squeeze(mean(v_w(:,:,:).*U(:,:,:),2));
        

   tau11_SGS_xz(:,:) = squeeze(mean(tau11_SGS(:,:,:),2));
   tau22_SGS_xz(:,:) = squeeze(mean(tau22_SGS(:,:,:),2));
   tau33_SGS_xz(:,:) = squeeze(mean(tau33_SGS(:,:,:),2));
   tau12_SGS_xz(:,:) = squeeze(mean(tau12_SGS(:,:,:),2));
   tau21_SGS_xz(:,:) = squeeze(mean(tau21_SGS(:,:,:),2));
   tau13_SGS_xz(:,:) = squeeze(mean(tau13_SGS(:,:,:),2));
   tau31_SGS_xz(:,:) = squeeze(mean(tau31_SGS(:,:,:),2));
   tau23_SGS_xz(:,:) = squeeze(mean(tau23_SGS(:,:,:),2));
   tau32_SGS_xz(:,:) = squeeze(mean(tau32_SGS(:,:,:),2));

   tau11_nu_xz(:,:) = squeeze(mean(tau11_nu(:,:,:),2));
   tau22_nu_xz(:,:) = squeeze(mean(tau22_nu(:,:,:),2));
   tau33_nu_xz(:,:) = squeeze(mean(tau33_nu(:,:,:),2));
   tau12_nu_xz(:,:) = squeeze(mean(tau12_nu(:,:,:),2));
   tau21_nu_xz(:,:) = squeeze(mean(tau21_nu(:,:,:),2));
   tau13_nu_xz(:,:) = squeeze(mean(tau13_nu(:,:,:),2));
   tau31_nu_xz(:,:) = squeeze(mean(tau31_nu(:,:,:),2));
   tau23_nu_xz(:,:) = squeeze(mean(tau23_nu(:,:,:),2));
   tau32_nu_xz(:,:) = squeeze(mean(tau32_nu(:,:,:),2));

   tau11_nu_J_xz(:,:) = squeeze(mean(tau11_nu_J(:,:,:),2));
   tau22_nu_J_xz(:,:) = squeeze(mean(tau22_nu_J(:,:,:),2));
   tau33_nu_J_xz(:,:) = squeeze(mean(tau33_nu_J(:,:,:),2));
   tau12_nu_J_xz(:,:) = squeeze(mean(tau12_nu_J(:,:,:),2));
   tau21_nu_J_xz(:,:) = squeeze(mean(tau21_nu_J(:,:,:),2));
   tau13_nu_J_xz(:,:) = squeeze(mean(tau13_nu_J(:,:,:),2));
   tau31_nu_J_xz(:,:) = squeeze(mean(tau31_nu_J(:,:,:),2));
   tau23_nu_J_xz(:,:) = squeeze(mean(tau23_nu_J(:,:,:),2));
   tau32_nu_J_xz(:,:) = squeeze(mean(tau32_nu_J(:,:,:),2));

    %TKE terms
    %Viscous & SGS dissipation
    epsilon_xz(:,:) = squeeze(mean(epsilon(:,:,:),2));
    epsilon_J_xz(:,:) = squeeze(mean(epsilon_J(:,:,:),2));
    
    epsilon_SGS_xz(:,:) = squeeze(mean(epsilon_SGS(:,:,:),2));
    
    S11_xz(:,:) = squeeze(mean(S11(:,:,:),2));
    S22_xz(:,:) = squeeze(mean(S22(:,:,:),2));
    S33_xz(:,:) = squeeze(mean(S33(:,:,:),2));
    S12_xz(:,:) = squeeze(mean(S12(:,:,:),2));
    S13_xz(:,:) = squeeze(mean(S13(:,:,:),2));
    S23_xz(:,:) = squeeze(mean(S23(:,:,:),2));

    %Viscous & SGS transport
    u_tau11_nu_xz(:,:) = squeeze(mean(u_w(:,:,:).*tau11_nu(:,:,:),2));
    v_tau21_nu_xz(:,:) = squeeze(mean(v_w(:,:,:).*tau21_nu(:,:,:),2));
    w_tau31_nu_xz(:,:) = squeeze(mean(  w(:,:,:).*tau31_nu(:,:,:),2));
    
    u_tau13_nu_xz(:,:) = squeeze(mean(u_w(:,:,:).*tau13_nu(:,:,:),2));
    v_tau23_nu_xz(:,:) = squeeze(mean(v_w(:,:,:).*tau23_nu(:,:,:),2));
    w_tau33_nu_xz(:,:) = squeeze(mean(  w(:,:,:).*tau33_nu(:,:,:),2));
    

    u_tau11_SGS_xz(:,:) = squeeze(mean(u_w(:,:,:).*tau11_SGS(:,:,:),2));
    v_tau21_SGS_xz(:,:) = squeeze(mean(v_w(:,:,:).*tau21_SGS(:,:,:),2));
    w_tau31_SGS_xz(:,:) = squeeze(mean(  w(:,:,:).*tau31_SGS(:,:,:),2));
    
    u_tau13_SGS_xz(:,:) = squeeze(mean(u_w(:,:,:).*tau13_SGS(:,:,:),2));
    v_tau23_SGS_xz(:,:) = squeeze(mean(v_w(:,:,:).*tau23_SGS(:,:,:),2));
    w_tau33_SGS_xz(:,:) = squeeze(mean(  w(:,:,:).*tau33_SGS(:,:,:),2));    
    
    
    %pressure transport
    pU_xz(:,:) = squeeze(mean(p_w(:,:,:).*U(:,:,:),2));
    pW_xz(:,:) = squeeze(mean(p_w(:,:,:).*W(:,:,:),2));
    
    %Convective transport
    uiuiU_xz(:,:) = squeeze(mean(u_w(:,:,:).*u_w(:,:,:).*U(:,:,:) ...
                                +v_w(:,:,:).*v_w(:,:,:).*U(:,:,:) ...
                                +  w(:,:,:).*  w(:,:,:).*U(:,:,:),2));

    uiuiW_xz(:,:) = squeeze(mean(u_w(:,:,:).*u_w(:,:,:).*W(:,:,:) ...
                                +v_w(:,:,:).*v_w(:,:,:).*W(:,:,:) ...
                                +  w(:,:,:).*  w(:,:,:).*W(:,:,:),2));
    

   %disp('reached')
   %perform phase shift in x-direction
     
   u_shifted   = zeros(Nz, Nx);
   u_w_shifted = zeros(Nz, Nx);
   v_shifted   = zeros(Nz, Nx);
   v_w_shifted = zeros(Nz, Nx);

   w_shifted = zeros(Nz, Nx);
     p_shifted = zeros(Nz, Nx);
   p_w_shifted = zeros(Nz, Nx);

     W_shifted = zeros(Nz, Nx);
     U_shifted = zeros(Nz, Nx);
   eta_shifted = zeros( 1, Nx);
   taup_13_shifted = zeros(Nz, Nx);
   taup_11_shifted = zeros(Nz, Nx);
   
     
   uW_shifted = zeros(Nz, Nx);
   uU_shifted = zeros(Nz, Nx);
   wU_shifted = zeros(Nz, Nx);
   wW_shifted = zeros(Nz, Nx);
   vU_shifted = zeros(Nz, Nx);
   vW_shifted = zeros(Nz, Nx);
   

   %tau_SGS_ij
    tau11_SGS_shifted= zeros(Nz,Nx);
    tau22_SGS_shifted= zeros(Nz,Nx);
    tau33_SGS_shifted= zeros(Nz,Nx);
    tau12_SGS_shifted= zeros(Nz,Nx);
    tau21_SGS_shifted= zeros(Nz,Nx);
    tau13_SGS_shifted= zeros(Nz,Nx);
    tau31_SGS_shifted= zeros(Nz,Nx);
    tau32_SGS_shifted= zeros(Nz,Nx);
    tau23_SGS_shifted= zeros(Nz,Nx);
    %tau_nu_ij
    tau11_nu_shifted= zeros(Nz,Nx);
    tau22_nu_shifted= zeros(Nz,Nx);
    tau33_nu_shifted= zeros(Nz,Nx);
    tau12_nu_shifted= zeros(Nz,Nx);
    tau21_nu_shifted= zeros(Nz,Nx);
    tau13_nu_shifted= zeros(Nz,Nx);
    tau31_nu_shifted= zeros(Nz,Nx);
    tau32_nu_shifted= zeros(Nz,Nx);
    tau23_nu_shifted= zeros(Nz,Nx);

    tau11_nu_J_shifted= zeros(Nz,Nx);
    tau22_nu_J_shifted= zeros(Nz,Nx);
    tau33_nu_J_shifted= zeros(Nz,Nx);
    tau12_nu_J_shifted= zeros(Nz,Nx);
    tau21_nu_J_shifted= zeros(Nz,Nx);
    tau13_nu_J_shifted= zeros(Nz,Nx);
    tau31_nu_J_shifted= zeros(Nz,Nx);
    tau32_nu_J_shifted= zeros(Nz,Nx);
    tau23_nu_J_shifted= zeros(Nz,Nx);

    uu_w_shifted = zeros(Nz, Nx);%TKE
    vv_w_shifted = zeros(Nz, Nx);%TKE
    ww_shifted = zeros(Nz, Nx);%TKE

    %TKE dissipation
    epsilon_shifted = zeros(Nz,Nx);
    epsilon_J_shifted = zeros(Nz,Nx);
    
    epsilon_SGS_shifted = zeros(Nz,Nx);
    S11_shifted = zeros(Nz,Nx);
    S22_shifted = zeros(Nz,Nx);
    S33_shifted = zeros(Nz,Nx);
    S12_shifted = zeros(Nz,Nx);
    S13_shifted = zeros(Nz,Nx);
    S23_shifted = zeros(Nz,Nx);
    
    %Viscous transport
    u_tau11_nu_shifted= zeros(Nz,Nx);
    v_tau21_nu_shifted= zeros(Nz,Nx);
    w_tau31_nu_shifted= zeros(Nz,Nx);

    u_tau13_nu_shifted= zeros(Nz,Nx);
    v_tau23_nu_shifted= zeros(Nz,Nx);
    w_tau33_nu_shifted= zeros(Nz,Nx);
    
    %SGS transport
    u_tau11_SGS_shifted= zeros(Nz,Nx);
    v_tau21_SGS_shifted= zeros(Nz,Nx);
    w_tau31_SGS_shifted= zeros(Nz,Nx);

    u_tau13_SGS_shifted= zeros(Nz,Nx);
    v_tau23_SGS_shifted= zeros(Nz,Nx);
    w_tau33_SGS_shifted= zeros(Nz,Nx);


    %pressure transport
    pU_shifted = zeros(Nz,Nx);
    pW_shifted = zeros(Nz,Nx);
    
    %convective transport
    uiuiU_shifted = zeros(Nz,Nx);
    uiuiW_shifted = zeros(Nz,Nx);


   for k = 1:Nz
            u_shifted(k, :) = phase_avg_shift(  u_xz(1:Nx, k)', kx, Nx, time, c_phase);
          u_w_shifted(k, :) = phase_avg_shift(u_w_xz(1:Nx, k)', kx, Nx, time, c_phase);
            v_shifted(k, :) = phase_avg_shift(  v_xz(1:Nx, k)', kx, Nx, time, c_phase);
          v_w_shifted(k, :) = phase_avg_shift(v_w_xz(1:Nx, k)', kx, Nx, time, c_phase);

          uu_w_shifted(k, :) = phase_avg_shift(uu_w_xz(1:Nx, k)', kx, Nx, time, c_phase);%TKE
          vv_w_shifted(k, :) = phase_avg_shift(vv_w_xz(1:Nx, k)', kx, Nx, time, c_phase);%TKE
            ww_shifted(k, :) = phase_avg_shift(  ww_xz(1:Nx, k)', kx, Nx, time, c_phase);%TKE

            w_shifted(k, :) = phase_avg_shift(  w_xz(1:Nx, k)', kx, Nx, time, c_phase);
            p_shifted(k, :) = phase_avg_shift(  p_xz(1:Nx, k)', kx, Nx, time, c_phase);
          p_w_shifted(k, :) = phase_avg_shift(p_w_xz(1:Nx, k)', kx, Nx, time, c_phase);
            W_shifted(k, :) = phase_avg_shift(  W_xz(1:Nx, k)', kx, Nx, time, c_phase); 
            U_shifted(k, :) = phase_avg_shift(  U_xz(1:Nx, k)', kx, Nx, time, c_phase);

        taup_13_shifted(k, :) = phase_avg_shift( taup_13_xz(1:Nx, k)', kx, Nx, time, c_phase);
        taup_11_shifted(k, :) = phase_avg_shift( taup_11_xz(1:Nx, k)', kx, Nx, time, c_phase);

        uW_shifted(k, :) = phase_avg_shift( uW_xz(1:Nx, k)', kx, Nx, time, c_phase);
        uU_shifted(k, :) = phase_avg_shift( uU_xz(1:Nx, k)', kx, Nx, time, c_phase);
        wW_shifted(k, :) = phase_avg_shift( wW_xz(1:Nx, k)', kx, Nx, time, c_phase);
        wU_shifted(k, :) = phase_avg_shift( wU_xz(1:Nx, k)', kx, Nx, time, c_phase);
        vW_shifted(k, :) = phase_avg_shift( vW_xz(1:Nx, k)', kx, Nx, time, c_phase);
        vU_shifted(k, :) = phase_avg_shift( vU_xz(1:Nx, k)', kx, Nx, time, c_phase);

       tau11_SGS_shifted(k, :) = phase_avg_shift( tau11_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau22_SGS_shifted(k, :) = phase_avg_shift( tau22_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau33_SGS_shifted(k, :) = phase_avg_shift( tau33_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau12_SGS_shifted(k, :) = phase_avg_shift( tau12_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau21_SGS_shifted(k, :) = phase_avg_shift( tau21_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau13_SGS_shifted(k, :) = phase_avg_shift( tau13_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau31_SGS_shifted(k, :) = phase_avg_shift( tau31_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau23_SGS_shifted(k, :) = phase_avg_shift( tau23_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau32_SGS_shifted(k, :) = phase_avg_shift( tau32_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);

       tau11_nu_shifted(k, :) = phase_avg_shift( tau11_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau22_nu_shifted(k, :) = phase_avg_shift( tau22_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau33_nu_shifted(k, :) = phase_avg_shift( tau33_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau12_nu_shifted(k, :) = phase_avg_shift( tau12_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau21_nu_shifted(k, :) = phase_avg_shift( tau21_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau13_nu_shifted(k, :) = phase_avg_shift( tau13_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau31_nu_shifted(k, :) = phase_avg_shift( tau31_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau23_nu_shifted(k, :) = phase_avg_shift( tau23_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau32_nu_shifted(k, :) = phase_avg_shift( tau32_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);

       tau11_nu_J_shifted(k, :) = phase_avg_shift( tau11_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau22_nu_J_shifted(k, :) = phase_avg_shift( tau22_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau33_nu_J_shifted(k, :) = phase_avg_shift( tau33_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau12_nu_J_shifted(k, :) = phase_avg_shift( tau12_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau21_nu_J_shifted(k, :) = phase_avg_shift( tau21_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau13_nu_J_shifted(k, :) = phase_avg_shift( tau13_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau31_nu_J_shifted(k, :) = phase_avg_shift( tau31_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau23_nu_J_shifted(k, :) = phase_avg_shift( tau23_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       tau32_nu_J_shifted(k, :) = phase_avg_shift( tau32_nu_J_xz(1:Nx, k)', kx, Nx, time, c_phase);

       epsilon_shifted(k, :) = phase_avg_shift( epsilon_xz(1:Nx, k)', kx, Nx, time, c_phase);
       epsilon_J_shifted(k, :) = phase_avg_shift( epsilon_J_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       epsilon_SGS_shifted(k, :) = phase_avg_shift( epsilon_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       S11_shifted(k, :) = phase_avg_shift( S11_xz(1:Nx, k)', kx, Nx, time, c_phase);
       S22_shifted(k, :) = phase_avg_shift( S22_xz(1:Nx, k)', kx, Nx, time, c_phase);
       S33_shifted(k, :) = phase_avg_shift( S33_xz(1:Nx, k)', kx, Nx, time, c_phase);
       S12_shifted(k, :) = phase_avg_shift( S12_xz(1:Nx, k)', kx, Nx, time, c_phase);
       S13_shifted(k, :) = phase_avg_shift( S13_xz(1:Nx, k)', kx, Nx, time, c_phase);
       S23_shifted(k, :) = phase_avg_shift( S23_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       u_tau11_nu_shifted(k, :) = phase_avg_shift( u_tau11_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       v_tau21_nu_shifted(k, :) = phase_avg_shift( v_tau21_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       w_tau31_nu_shifted(k, :) = phase_avg_shift( w_tau31_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       u_tau13_nu_shifted(k, :) = phase_avg_shift( u_tau13_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       v_tau23_nu_shifted(k, :) = phase_avg_shift( v_tau23_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       w_tau33_nu_shifted(k, :) = phase_avg_shift( w_tau33_nu_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       u_tau11_SGS_shifted(k, :) = phase_avg_shift( u_tau11_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       v_tau21_SGS_shifted(k, :) = phase_avg_shift( v_tau21_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       w_tau31_SGS_shifted(k, :) = phase_avg_shift( w_tau31_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       u_tau13_SGS_shifted(k, :) = phase_avg_shift( u_tau13_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       v_tau23_SGS_shifted(k, :) = phase_avg_shift( v_tau23_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);
       w_tau33_SGS_shifted(k, :) = phase_avg_shift( w_tau33_SGS_xz(1:Nx, k)', kx, Nx, time, c_phase);


       pU_shifted(k, :) = phase_avg_shift( pU_xz(1:Nx, k)', kx, Nx, time, c_phase);
       pW_shifted(k, :) = phase_avg_shift( pW_xz(1:Nx, k)', kx, Nx, time, c_phase);

       uiuiU_shifted(k, :) = phase_avg_shift( uiuiU_xz(1:Nx, k)', kx, Nx, time, c_phase);
       uiuiW_shifted(k, :) = phase_avg_shift( uiuiW_xz(1:Nx, k)', kx, Nx, time, c_phase);
       
       
   end 
   %storing surface pressure as a function of time
   p_shifted_surf = zeros(Nx,1);
   p_w_shifted_surf = zeros(Nx,1);
   p_shifted_surf(:,1) = p_shifted(1,:)-mean(p_shifted(1,:));
   p_w_shifted_surf(:,1) = p_w_shifted(1,:)-mean(p_w_shifted(1,:));
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %phase average
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
                 eta_shifted = phase_avg_shift(squeeze(mean(eta(1:Nx, :),2))', kx, Nx, time, c_phase);
       eta_phase_avg(1:Nx,:) = eta_phase_avg(1:Nx,:) + (1 / Nt) * eta_shifted';
         u_phase_avg(1:Nx,:) =   u_phase_avg(1:Nx,:) + (1 / Nt) *   u_shifted';
       u_w_phase_avg(1:Nx,:) = u_w_phase_avg(1:Nx,:) + (1 / Nt) * u_w_shifted';
         v_phase_avg(1:Nx,:) =   v_phase_avg(1:Nx,:) + (1 / Nt) *   v_shifted';
       v_w_phase_avg(1:Nx,:) = v_w_phase_avg(1:Nx,:) + (1 / Nt) * v_w_shifted';
         w_phase_avg(1:Nx,:) =   w_phase_avg(1:Nx,:) + (1 / Nt) *   w_shifted';
         p_phase_avg(1:Nx,:) =   p_phase_avg(1:Nx,:) + (1 / Nt) *   p_shifted';
       p_w_phase_avg(1:Nx,:) = p_w_phase_avg(1:Nx,:) + (1 / Nt) * p_w_shifted'; 
         W_phase_avg(1:Nx,:) =   W_phase_avg(1:Nx,:) + (1 / Nt) *   W_shifted';
         U_phase_avg(1:Nx,:) =   U_phase_avg(1:Nx,:) + (1 / Nt) *   U_shifted';
   taup_13_phase_avg(1:Nx,:) = taup_13_phase_avg(1:Nx,:) + (1 / Nt) * taup_13_shifted';
   taup_11_phase_avg(1:Nx,:) = taup_11_phase_avg(1:Nx,:) + (1 / Nt) * taup_11_shifted';
        
   uW_phase_avg(1:Nx,:) = uW_phase_avg(1:Nx,:) + (1 / Nt) * uW_shifted';
   uU_phase_avg(1:Nx,:) = uU_phase_avg(1:Nx,:) + (1 / Nt) * uU_shifted';
   wW_phase_avg(1:Nx,:) = wW_phase_avg(1:Nx,:) + (1 / Nt) * wW_shifted';
   wU_phase_avg(1:Nx,:) = wU_phase_avg(1:Nx,:) + (1 / Nt) * wU_shifted';
   vW_phase_avg(1:Nx,:) = vW_phase_avg(1:Nx,:) + (1 / Nt) * vW_shifted';
   vU_phase_avg(1:Nx,:) = vU_phase_avg(1:Nx,:) + (1 / Nt) * vU_shifted';

   tau11_SGS_phase_avg(1:Nx,:) = tau11_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau11_SGS_shifted';
   tau22_SGS_phase_avg(1:Nx,:) = tau22_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau22_SGS_shifted';
   tau33_SGS_phase_avg(1:Nx,:) = tau33_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau33_SGS_shifted';
   tau12_SGS_phase_avg(1:Nx,:) = tau12_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau12_SGS_shifted';
   tau21_SGS_phase_avg(1:Nx,:) = tau21_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau21_SGS_shifted';
   tau13_SGS_phase_avg(1:Nx,:) = tau13_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau13_SGS_shifted';
   tau31_SGS_phase_avg(1:Nx,:) = tau31_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau31_SGS_shifted';
   tau23_SGS_phase_avg(1:Nx,:) = tau23_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau23_SGS_shifted';
   tau32_SGS_phase_avg(1:Nx,:) = tau32_SGS_phase_avg(1:Nx,:) + (1 / Nt) * tau32_SGS_shifted';
   
   tau11_nu_phase_avg(1:Nx,:) = tau11_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau11_nu_shifted';
   tau22_nu_phase_avg(1:Nx,:) = tau22_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau22_nu_shifted';
   tau33_nu_phase_avg(1:Nx,:) = tau33_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau33_nu_shifted';
   tau12_nu_phase_avg(1:Nx,:) = tau12_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau12_nu_shifted';
   tau21_nu_phase_avg(1:Nx,:) = tau21_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau21_nu_shifted';
   tau13_nu_phase_avg(1:Nx,:) = tau13_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau13_nu_shifted';
   tau31_nu_phase_avg(1:Nx,:) = tau31_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau31_nu_shifted';
   tau23_nu_phase_avg(1:Nx,:) = tau23_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau23_nu_shifted';
   tau32_nu_phase_avg(1:Nx,:) = tau32_nu_phase_avg(1:Nx,:) + (1 / Nt) * tau32_nu_shifted';

   tau11_nu_J_phase_avg(1:Nx,:) = tau11_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau11_nu_J_shifted';
   tau22_nu_J_phase_avg(1:Nx,:) = tau22_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau22_nu_J_shifted';
   tau33_nu_J_phase_avg(1:Nx,:) = tau33_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau33_nu_J_shifted';
   tau12_nu_J_phase_avg(1:Nx,:) = tau12_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau12_nu_J_shifted';
   tau21_nu_J_phase_avg(1:Nx,:) = tau21_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau21_nu_J_shifted';
   tau13_nu_J_phase_avg(1:Nx,:) = tau13_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau13_nu_J_shifted';
   tau31_nu_J_phase_avg(1:Nx,:) = tau31_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau31_nu_J_shifted';
   tau23_nu_J_phase_avg(1:Nx,:) = tau23_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau23_nu_J_shifted';
   tau32_nu_J_phase_avg(1:Nx,:) = tau32_nu_J_phase_avg(1:Nx,:) + (1 / Nt) * tau32_nu_J_shifted';

   %TKE dissipation
   epsilon_phase_avg(1:Nx,:) = epsilon_phase_avg(1:Nx,:) + (1 / Nt) * epsilon_shifted';
   epsilon_J_phase_avg(1:Nx,:) = epsilon_J_phase_avg(1:Nx,:) + (1 / Nt) * epsilon_J_shifted';
   
   epsilon_SGS_phase_avg(1:Nx,:) = epsilon_SGS_phase_avg(1:Nx,:) + (1 / Nt) * epsilon_SGS_shifted';
   
   S11_phase_avg(1:Nx,:) = S11_phase_avg(1:Nx,:) + (1 / Nt) * S11_shifted';
   S22_phase_avg(1:Nx,:) = S22_phase_avg(1:Nx,:) + (1 / Nt) * S22_shifted';
   S33_phase_avg(1:Nx,:) = S33_phase_avg(1:Nx,:) + (1 / Nt) * S33_shifted';
   S12_phase_avg(1:Nx,:) = S12_phase_avg(1:Nx,:) + (1 / Nt) * S12_shifted';
   S13_phase_avg(1:Nx,:) = S13_phase_avg(1:Nx,:) + (1 / Nt) * S13_shifted';
   S23_phase_avg(1:Nx,:) = S23_phase_avg(1:Nx,:) + (1 / Nt) * S23_shifted';

   %TKE   
   uu_w_phase_avg(1:Nx,:) = uu_w_phase_avg(1:Nx,:) + (1/Nt) * uu_w_shifted';
   vv_w_phase_avg(1:Nx,:) = vv_w_phase_avg(1:Nx,:) + (1/Nt) * vv_w_shifted';
     ww_phase_avg(1:Nx,:) =   ww_phase_avg(1:Nx,:) + (1/Nt) *   ww_shifted';
   
   %Viscous & SGS transport
   u_tau11_nu_phase_avg(1:Nx,:) =   u_tau11_nu_phase_avg(1:Nx,:) + (1/Nt) *   u_tau11_nu_shifted';
   v_tau21_nu_phase_avg(1:Nx,:) =   v_tau21_nu_phase_avg(1:Nx,:) + (1/Nt) *   v_tau21_nu_shifted';
   w_tau31_nu_phase_avg(1:Nx,:) =   w_tau31_nu_phase_avg(1:Nx,:) + (1/Nt) *   w_tau31_nu_shifted';

   u_tau13_nu_phase_avg(1:Nx,:) =   u_tau13_nu_phase_avg(1:Nx,:) + (1/Nt) *   u_tau13_nu_shifted';
   v_tau23_nu_phase_avg(1:Nx,:) =   v_tau23_nu_phase_avg(1:Nx,:) + (1/Nt) *   v_tau23_nu_shifted';
   w_tau33_nu_phase_avg(1:Nx,:) =   w_tau33_nu_phase_avg(1:Nx,:) + (1/Nt) *   w_tau33_nu_shifted';
   
   u_tau11_SGS_phase_avg(1:Nx,:) =   u_tau11_SGS_phase_avg(1:Nx,:) + (1/Nt) *   u_tau11_SGS_shifted';
   v_tau21_SGS_phase_avg(1:Nx,:) =   v_tau21_SGS_phase_avg(1:Nx,:) + (1/Nt) *   v_tau21_SGS_shifted';
   w_tau31_SGS_phase_avg(1:Nx,:) =   w_tau31_SGS_phase_avg(1:Nx,:) + (1/Nt) *   w_tau31_SGS_shifted';

   u_tau13_SGS_phase_avg(1:Nx,:) =   u_tau13_SGS_phase_avg(1:Nx,:) + (1/Nt) *   u_tau13_SGS_shifted';
   v_tau23_SGS_phase_avg(1:Nx,:) =   v_tau23_SGS_phase_avg(1:Nx,:) + (1/Nt) *   v_tau23_SGS_shifted';
   w_tau33_SGS_phase_avg(1:Nx,:) =   w_tau33_SGS_phase_avg(1:Nx,:) + (1/Nt) *   w_tau33_SGS_shifted'; 

   %pressure transport
   pU_phase_avg(1:Nx,:) =  pU_phase_avg(1:Nx,:) + (1/Nt) * pU_shifted';
   pW_phase_avg(1:Nx,:) =  pW_phase_avg(1:Nx,:) + (1/Nt) * pW_shifted';

   %Convective transport
   uiuiU_phase_avg(1:Nx,:) = uiuiU_phase_avg(1:Nx,:) + (1/Nt) * uiuiU_shifted';
   uiuiW_phase_avg(1:Nx,:) = uiuiW_phase_avg(1:Nx,:) + (1/Nt) * uiuiW_shifted';
   
   %disp([counter,min(u_w_phase_avg(:)),max(u_w_phase_avg(:))]);
   counter = counter + 1;


   tau13_mean_array=zeros(Nz,5);
   %tau13_tot = tau13_visc + tau13_p + tau13_SGS + tau13_turb  + tau13_wave;
   tau13_mean_array(1:Nz,1)=squeeze(mean(tau13_nu_shifted(1:Nz,1:Nx),2));%tau_visc
   tau13_mean_array(1:Nz,2)=squeeze(mean(taup_13_shifted(1:Nz,1:Nx),2));%tau_p
   tau13_mean_array(1:Nz,3)=squeeze(mean(tau13_SGS_shifted(1:Nz,1:Nx),2));%tau_SGS
   tau13_mean_array(1:Nz,4)=+squeeze(mean(uW_shifted - u_w_shifted.*W_shifted,2));%tau_turb

   for k=1:Nz
       tau13_mean_array(k,5)=+squeeze(mean((u_w_shifted(k,:) - mean(u_w_shifted(k,:),2)) ...
                                             .*(  W_shifted(k,:) - mean(W_shifted(k,:),2)),2));%tau_wave
   end

   save(sprintf('inst_data_all_02_c2_%0.0d.mat',n), ...
         'zw','zz','x','time', ...
         'tau13_mean_array', ...
         'p_shifted_surf', ...
         'p_w_shifted_surf');
   toc
end

%tau_turb
tau13_phase_avg(:,:) = +( uW_phase_avg(1:Nx,:) - u_w_phase_avg(1:Nx,:).*W_phase_avg(:,:) );
tau31_phase_avg(:,:) = +( wU_phase_avg(1:Nx,:) -   w_phase_avg(1:Nx,:).*U_phase_avg(:,:) );
tau11_phase_avg(:,:) = +( uU_phase_avg(1:Nx,:) - u_w_phase_avg(1:Nx,:).*U_phase_avg(:,:) );
tau33_phase_avg(:,:) = +( wW_phase_avg(1:Nx,:) -   w_phase_avg(1:Nx,:).*W_phase_avg(:,:) );

%tau_wave
for k=1:Nz
    tau13_wave_phase_avg(:,k) = +(u_w_phase_avg(:,k)-mean(u_w_phase_avg(:,k),1)) ...
                               .*(W_phase_avg(:,k)-mean(W_phase_avg(:,k),1));
    tau31_wave_phase_avg(:,k) = +(w_phase_avg(:,k)-mean(w_phase_avg(:,k),1)) ...
                               .*(U_phase_avg(:,k)-mean(U_phase_avg(:,k),1));
    tau11_wave_phase_avg(:,k) = +(u_w_phase_avg(:,k)-mean(u_w_phase_avg(:,k),1)) ...
                               .*(U_phase_avg(:,k)-mean(U_phase_avg(:,k),1));
    tau33_wave_phase_avg(:,k) = +(w_phase_avg(:,k)-mean(w_phase_avg(:,k),1)) ...
                               .*(W_phase_avg(:,k)-mean(W_phase_avg(:,k),1));
    
    
end

%TKE
TKE(:,:) = 0.5*((uu_w_phase_avg - u_w_phase_avg.*u_w_phase_avg) + ...
                (vv_w_phase_avg - v_w_phase_avg.*v_w_phase_avg) + ...
                (  ww_phase_avg -   w_phase_avg.*  w_phase_avg)); 
%TKE dissipation
epsilon_phase_avg = epsilon_phase_avg - 2*nu*(S11_phase_avg.*S11_phase_avg + ...
                                              S22_phase_avg.*S22_phase_avg + ...
                                              S33_phase_avg.*S33_phase_avg + ...
                                            2*S12_phase_avg.*S12_phase_avg + ...
                                            2*S13_phase_avg.*S13_phase_avg + ...
                                            2*S23_phase_avg.*S23_phase_avg);

epsilon_J_phase_avg = epsilon_J_phase_avg - (tau11_nu_J_phase_avg.*S11_phase_avg + ...
                                             tau22_nu_J_phase_avg.*S22_phase_avg + ...
                                             tau33_nu_J_phase_avg.*S33_phase_avg + ...
                                           2*tau12_nu_J_phase_avg.*S12_phase_avg + ...
                                           2*tau13_nu_J_phase_avg.*S13_phase_avg + ...
                                           2*tau23_nu_J_phase_avg.*S23_phase_avg);


epsilon_SGS_phase_avg = epsilon_SGS_phase_avg - ( tau11_SGS_phase_avg.*S11_phase_avg ...
                                                 +tau22_SGS_phase_avg.*S22_phase_avg ...
                                                 +tau33_SGS_phase_avg.*S33_phase_avg ...
                                                 +tau12_SGS_phase_avg.*S12_phase_avg ...
                                                 +tau13_SGS_phase_avg.*S13_phase_avg ...
                                                 +tau23_SGS_phase_avg.*S23_phase_avg ...
                                                 +tau21_SGS_phase_avg.*S12_phase_avg ...
                                                 +tau31_SGS_phase_avg.*S13_phase_avg ...
                                                 +tau32_SGS_phase_avg.*S23_phase_avg );


%Pressure transport
pp_Up_TKE = pU_phase_avg - p_w_phase_avg.*U_phase_avg;
pp_Wp_TKE = pW_phase_avg - p_w_phase_avg.*W_phase_avg;

%Viscous transport
upi_taup_nu_i1_TKE = u_tau11_nu_phase_avg - u_w_phase_avg.*tau11_nu_phase_avg + ...
                     v_tau21_nu_phase_avg - v_w_phase_avg.*tau21_nu_phase_avg + ...
                     w_tau31_nu_phase_avg -   w_phase_avg.*tau31_nu_phase_avg;

upi_taup_nu_i3_TKE = u_tau13_nu_phase_avg - u_w_phase_avg.*tau13_nu_phase_avg + ...
                     v_tau23_nu_phase_avg - v_w_phase_avg.*tau23_nu_phase_avg + ...
                     w_tau33_nu_phase_avg -   w_phase_avg.*tau33_nu_phase_avg;

upi_taup_SGS_i1_TKE = u_tau11_SGS_phase_avg - u_w_phase_avg.*tau11_SGS_phase_avg + ...
                      v_tau21_SGS_phase_avg - v_w_phase_avg.*tau21_SGS_phase_avg + ...
                      w_tau31_SGS_phase_avg -   w_phase_avg.*tau31_SGS_phase_avg;

upi_taup_SGS_i3_TKE = u_tau13_SGS_phase_avg - u_w_phase_avg.*tau13_SGS_phase_avg + ...
                      v_tau23_SGS_phase_avg - v_w_phase_avg.*tau23_SGS_phase_avg + ...
                      w_tau33_SGS_phase_avg -   w_phase_avg.*tau33_SGS_phase_avg;
%Convective transport
uiuiU_TKE = 0.5*(uiuiU_phase_avg ...
            -(uu_w_phase_avg+vv_w_phase_avg+ww_phase_avg).*U_phase_avg ...
            -2*u_w_phase_avg.*(uU_phase_avg-u_w_phase_avg.*U_phase_avg) ...
            -2*v_w_phase_avg.*(vU_phase_avg-v_w_phase_avg.*U_phase_avg) ...
            -  2*w_phase_avg.*(wU_phase_avg-  w_phase_avg.*U_phase_avg));
            
uiuiW_TKE = 0.5*(uiuiW_phase_avg ...
            -(uu_w_phase_avg+vv_w_phase_avg+ww_phase_avg).*W_phase_avg ...
            -2*u_w_phase_avg.*(uW_phase_avg-u_w_phase_avg.*W_phase_avg) ...
            -2*v_w_phase_avg.*(vW_phase_avg-v_w_phase_avg.*W_phase_avg) ...
            -  2*w_phase_avg.*(wW_phase_avg-  w_phase_avg.*W_phase_avg));

save('phase_averaged_data_all_02_c2.mat','zw','zz', ...
     'u_phase_avg', ...
     'u_w_phase_avg', ...
     'v_phase_avg',...
     'v_w_phase_avg',...
     'w_phase_avg', ...
     'p_phase_avg', ...
     'p_w_phase_avg', ...
     'eta_phase_avg', ...
     'U_phase_avg', ...
     'W_phase_avg', ...
     'uU_phase_avg', ...
     'uW_phase_avg', ...
     'vU_phase_avg', ...
     'vW_phase_avg', ...
     'wU_phase_avg', ...
     'wW_phase_avg', ...
     'pU_phase_avg', ...
     'pW_phase_avg', ...
     'taup_13_phase_avg', ...%tau_p
     'taup_11_phase_avg', ...
     'tau11_SGS_phase_avg', ...%tau_SGS
     'tau22_SGS_phase_avg',...
     'tau33_SGS_phase_avg',...
     'tau12_SGS_phase_avg',...
     'tau21_SGS_phase_avg',...
     'tau13_SGS_phase_avg',...
     'tau31_SGS_phase_avg',...
     'tau23_SGS_phase_avg',...
     'tau32_SGS_phase_avg',...
     'tau11_nu_phase_avg', ...%tau_nu
     'tau22_nu_phase_avg',...
     'tau33_nu_phase_avg',...
     'tau12_nu_phase_avg',...
     'tau21_nu_phase_avg',...
     'tau13_nu_phase_avg',...
     'tau31_nu_phase_avg',...
     'tau23_nu_phase_avg',...
     'tau32_nu_phase_avg',...
     'tau13_phase_avg',... %tau_turb
     'tau31_phase_avg',...
     'tau11_phase_avg',...
     'tau33_phase_avg',...
     'tau13_wave_phase_avg',...%tau_wave 
     'tau31_wave_phase_avg',...
     'tau11_wave_phase_avg',...
     'tau33_wave_phase_avg', ...
     'uu_w_phase_avg', ...%TKE
     'vv_w_phase_avg', ...
     'ww_phase_avg', ...
     'epsilon_phase_avg',...%viscous dissipation
     'epsilon_SGS_phase_avg',...%SGS dissipation
     'S11_phase_avg',...
     'S22_phase_avg',...
     'S33_phase_avg',...
     'S12_phase_avg',...
     'S13_phase_avg',...
     'S23_phase_avg',...
     'uiuiU_phase_avg',...%convective
     'uiuiW_phase_avg');


save('TKE_budget_terms_02_c2.mat','zw','zz', ...
             'TKE', ...
             'epsilon_phase_avg', ... %viscous dissipation
             'epsilon_J_phase_avg', ... %viscous dissipation-J             
             'epsilon_SGS_phase_avg', ...%SGS dissipation
             'uiuiW_TKE','uiuiU_TKE', ...%Convective transport
             'upi_taup_SGS_i1_TKE','upi_taup_SGS_i3_TKE', ...%SGS transport
             'upi_taup_nu_i3_TKE','upi_taup_nu_i1_TKE', ...%Viscous transport
             'pp_Up_TKE','pp_Wp_TKE'); %pressure transport 


%02_c2-15980000000
%03_c7-13740000000
%04_c15-13720000000
%05_c25-17520000000
%06_c0-17560000000
%07_c-2-16200000000
%08_c-7-13360000000
%09_c-15-13800000000
%10_c-25-15400000000

% 
% n_end=[15980000000,13740000000,13720000000, ...
%        17520000000,17560000000,16200000000, ...
%        13360000000,13800000000,15400000000];
