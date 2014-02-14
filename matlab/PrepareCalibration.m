function [A OT usedFrames]=PrepareCalibration(W,I,imHeight,imWidth,k_clusters)
%function [A OT usedFrames]=PrepareCalibration(W,I,imHeight,imWidth)
%Performs camera calibration using the world points W, image points I and
%distortion model 'model'. imHeight and imWidth == size of the original
%image (used to initialize principal point).
% size(W)=(2*nFrames,nPoints), size(I)=(2*nFrames,nPoints)
% Returns struct calib and the min reprojection error using this calibration.
% Based on a heavily modified version of
% init_intrinsic_param.m/compute_extrinsic.m from J. Bouguet
% http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html


c_init = [imWidth;imHeight]/2 - 0.5; % initialize at the center of the image
% matrix that subtract the principal point:
Sub_cc = [1 0 -c_init(1);0 1 -c_init(2);0 0 1];

% Compute explicitely the focal length using all the (mutually orthogonal) vanishing points
% note: The vanishing points are hidden in the planar collineations H_kk

usedFrames=find((sum(I(1:2:end,:)==-1,2)+sum(W(1:2:end,:)==-1,2))==0);
nValids=length(usedFrames);
V=zeros(2*nValids,2);
b=zeros(2*nValids,1);
usedLines=1;
for f=usedFrames'
    curI=squeeze(I(((2*(f-1))+1):(2*f),:));
    curW=squeeze(W(((2*(f-1))+1):(2*f),:));
    cal(f).H=ComputeHomography(curW,curI);
    [A_kk b_kk]=extractVanishingPoints(cal(f).H);
    V((2*(usedLines-1)+1):(2*usedLines),:)=A_kk;
    b((2*(usedLines-1)+1):(2*usedLines),:)=b_kk;
    usedLines=usedLines+1;
end

f_init = sqrt(abs(1./(((V'*V)\(V'))*b)));
% Initial estimate for intrinsic parameters
A = [f_init(1) 0 c_init(1);0 f_init(2) c_init(2); 0 0 1];


%% estimating extrinsic parameters

allOT=zeros(nValids,6);
corresp=zeros(nValids,1);
frErrors=zeros(nValids,1);
DEBUG_PLOT=0;
cF=1;
for f=usedFrames'
    curI=squeeze(I(((2*(f-1))+1):(2*f),:));
    curW=squeeze(W(((2*(f-1))+1):(2*f),:));
    [cal(f).o cal(f).T]=computeExtrinsic(f,curI,curW,0);
    frErrors(cF)=frameError(curI,curW,[f_init; c_init; 0],[cal(f).o; cal(f).T]);
    allOT(cF,:)=[cal(f).o;cal(f).T];
    corresp(cF)=f;
    cF=cF+1;
end

IDX=kmeans(allOT,k_clusters,'emptyaction','drop');
selected_idx=[];
cF=1;
for i=1:k_clusters
    tErr=frErrors;
    tErr(IDX~=i)=NaN;
    [~,cIdx]=max(tErr);
    if (~isempty(cIdx))
        selected_idx=[selected_idx;corresp(cIdx)]; %#ok<AGROW>
    end
end
usedFrames=sort(selected_idx);
nValids=length(usedFrames);
OT=zeros(nValids,6);

cF=1;
for f=usedFrames'
    curI=squeeze(I(((2*(f-1))+1):(2*f),:));
    curW=squeeze(W(((2*(f-1))+1):(2*f),:));
    [cal(f).o cal(f).T]=computeExtrinsic(f,curI,curW,1);
    OT(cF,:)=[cal(f).o;cal(f).T];
    cF=cF+1;
end




    function [cO cT]=computeExtrinsic(curF,curI,curW,opt)
        X_kk=[curW ; zeros(1,size(W,2))];
        xn = normalize_pixel(curI,f_init,c_init);
        Np = size(xn,2);
        
        % Check for planarity of the structure:
        X_mean = mean(X_kk,2);
        R_transform = eye(3) ;
        T_transform = -(R_transform)*X_mean;
        X_new = R_transform*X_kk + T_transform*ones(1,Np);
        % Compute the planar homography:
        H = compute_homography(xn,X_new(1:2,:));
        % De-embed the motion parameters from the homography:
        sc = mean([norm(H(:,1));norm(H(:,2))]);
        H = H/sc;
        
        u1 = H(:,1);
        u1 = u1 / norm(u1);
        u2 = H(:,2) - dot(u1,H(:,2)) * u1;
        u2 = u2 / norm(u2);
        u3 = cross(u1,u2);
        RRR = [u1 u2 u3];
        Rckk = rodrigues(rodrigues(RRR));
        Tckk = H(:,3);
        
        %If Xc = Rckk * X_new + Tckk, then Xc = Rckk * R_transform * X_kk + Tckk + T_transform
        cT = Tckk + Rckk* T_transform;
        Rckk = Rckk * R_transform;
        cO = rodrigues(Rckk);
        if ((nargin>2)&&(opt==1))
            DEBUG_PLOT=0;
            nX=fminsearch(@fcn_opt_OT,[cO; cT]);
            cO=nX(1:3);
            cT=nX(4:6);
        end
        function vErr=fcn_opt_OT(xIn)
            vErr=frameError(curI, curW,...
                [A(1,1) A(2,2) A(1,3) A(2,3) 0], xIn);
        end
        
    end
    function [A_r b_r]=extractVanishingPoints(Hkk)
        Hkk = Sub_cc * Hkk;
        
        % Extract vanishing points (direct and diagonals):
        
        V_hori_pix = Hkk(:,1);
        V_vert_pix = Hkk(:,2);
        V_diag1_pix = (Hkk(:,1)+Hkk(:,2))/2;
        V_diag2_pix = (Hkk(:,1)-Hkk(:,2))/2;
        
        V_hori_pix = V_hori_pix/norm(V_hori_pix);
        V_vert_pix = V_vert_pix/norm(V_vert_pix);
        V_diag1_pix = V_diag1_pix/norm(V_diag1_pix);
        V_diag2_pix = V_diag2_pix/norm(V_diag2_pix);
        
        a1 = V_hori_pix(1);
        b1 = V_hori_pix(2);
        c1 = V_hori_pix(3);
        
        a2 = V_vert_pix(1);
        b2 = V_vert_pix(2);
        c2 = V_vert_pix(3);
        
        a3 = V_diag1_pix(1);
        b3 = V_diag1_pix(2);
        c3 = V_diag1_pix(3);
        
        a4 = V_diag2_pix(1);
        b4 = V_diag2_pix(2);
        c4 = V_diag2_pix(3);
        
        A_r = [a1*a2  b1*b2;
            a3*a4  b3*b4];
        
        b_r = -[c1*c2;c3*c4];
    end
    function ferr=frameError(cI, cW, curIntrinsic, curOT)
        RT=[rodrigues(curOT(1:3)) curOT(4:6,1)];
        curA=[curIntrinsic(1) curIntrinsic(5) curIntrinsic(3);...
            0 curIntrinsic(2) curIntrinsic(4);...
            0 0 1];
        
        X=RT*[cW;zeros(1,size(cW,2));ones(1,size(cW,2))];
        X=X(1:2,:)./repmat(X(3,:),[2 1]);
        xl=X;
        
        if ((size(xl,2)~=size(X,2))||(size(xl,1)~=2))
            xl=NaN;
        else
            xl=curA*[xl;ones(1,size(xl,2))];
            xl=xl(1:2,:)./repmat(xl(3,:),[2 1]);
        end
        ferr=sum(sqrt(sum((xl-cI).^2)));
        if (DEBUG_PLOT==1)
            %figure(10),hold off,
            figure
            plot(cI(1,:),cI(2,:),'o');
            hold on, grid on
            plot(xl(1,:),xl(2,:),'r+');
            legend('Original value','Reprojected');
            title(sprintf('Error : %g',ferr));
        end
        
    end
    function [xn] = normalize_pixel(x_kk,fc,cc)
        %Computes the normalized coordinates xn given the pixel coordinates x_kk
        %and the intrinsic camera parameters fc, cc and
        %
        %INPUT: x_kk: Feature locations on the images
        %       fc: Camera focal length
        %       cc: Principal point coordinates
        %
        %OUTPUT: xn: Normalized feature locations on the image plane (a 2XN matrix)
        %
        %Important functions called within that program:
        %
        %comp_distortion_oulu: undistort pixel coordinates.
        
        alpha_c = 0;
        kc = [0;0;0;0;0];
        
        
        % First: Subtract principal point, and divide by the focal length:
        x_distort = [(x_kk(1,:) - cc(1))/fc(1);(x_kk(2,:) - cc(2))/fc(2)];
        
        % Second: undo skew
        x_distort(1,:) = x_distort(1,:) - alpha_c * x_distort(2,:);
        
        if norm(kc) ~= 0,
            % Third: Compensate for lens distortion:
            xn = comp_distortion_oulu(x_distort,kc);
        else
            xn = x_distort;
        end;
    end

end

