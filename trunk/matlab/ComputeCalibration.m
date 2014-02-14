function error=ComputeCalibration(W,I,model,usedFrames,A,OT)
%function [calib error]=ComputeCalibration(W,I,imHeight,imWidth,model)
%Performs camera calibration using the world points W, image points I and
%distortion model 'model'. imHeight and imWidth == size of the original
%image (used to initialize principal point).
% size(W)=(2*nFrames,nPoints), size(I)=(2*nFrames,nPoints)
% Returns struct calib and the min reprojection error using this calibration.
% Based on a heavily modified version of
% init_intrinsic_param.m/compute_extrinsic.m from J. Bouguet
% http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html


maxTime1=0.2;
maxTime2=20;

vars=unique(regexp(model,'x[0-9]+','match'));
coeffs=unique(regexp(model,'d[0-9]+','match'));
nConst=size(vars,2);
nCoeff=size(coeffs,2);
for i=1:nConst
    model=strrep(model,vars{i},sprintf('x(%d)',i));
end
cI=1;
for i=(nConst+1):(nConst+nCoeff)
    model=strrep(model,coeffs{cI},sprintf('x(%d)',i));
    cI=cI+1;
end

nConst=nConst+nCoeff;

xx=composeOptVector(OT);
xNorm=xx;
xNorm(xNorm<=1)=1;

DEBUG_PLOT=0;
if (nConst>0)
    opts = saoptimset('TimeLimit',1); %fast check so we do not really evaluate crazy models
    [xdist,error] = simulannealbnd(@fcn_opt_dist,zeros(nConst,1), [], [], opts);
else
    xdist=[];
    error=0;
end

if (isnan(error))
    error=Inf;
else
    xm=[xdist; ones(size(xx,1)-nConst,1)];
    opts = optimset('MaxFunEvals',1E8);%'TimeLimit',maxTime2);%,'Display','iter','PlotFcns',@saplotbestf);%,'TypicalX',xTyp);
    [xSa,error,exitflag,output] = fminsearch(@fcn_opt,xx,opts);%, [], [], opts);[x,fval,exitflag,output]
    error=error/size(OT,1);
end


    function vErr=fcn_opt_dist(xIn)
        vErr=zeros(length(usedFrames),1);
        cF=1;
        for ii=usedFrames'
            ccI=squeeze(I(((2*(ii-1))+1):(2*ii),:));
            ccW=squeeze(W(((2*(ii-1))+1):(2*ii),:));
            if ((~any(ccI(1,:)==-1))&&(~any(ccW(1,:)==-1)))
                vErr(cF)=frameError(ccI, ccW,...
                    xx(nConst+1:nConst+5),...
                    xIn(1:nConst),...
                    xx((5+nConst+(6*(cF-1)+1)):(5+nConst+6*cF)));
                cF=cF+1;
            end
        end
        vErr=sum(vErr);
    end
    function vErr=fcn_opt(xIn)
        %xIn=xIn.*xNorm;
        vErr=zeros(length(usedFrames),1);
        cF=1;
        for ii=usedFrames'
            ccI=squeeze(I(((2*(ii-1))+1):(2*ii),:));
            ccW=squeeze(W(((2*(ii-1))+1):(2*ii),:));
            if ((~any(ccI(1,:)==-1))&&(~any(ccW(1,:)==-1)))
                vErr(cF)=frameError(ccI, ccW,...
                    xIn(nConst+1:nConst+5),...
                    xIn(1:nConst),...
                    xIn((5+nConst+(6*(cF-1)+1)):(5+nConst+6*cF)));
                cF=cF+1;
            end
        end
        vErr=sum(vErr);
    end
    function ferr=frameError(cI, cW, curIntrinsic, curDist, curOT)
        RT=[rodrigues(curOT(1:3)) curOT(4:6,1)];
        curA=[curIntrinsic(1) curIntrinsic(5) curIntrinsic(3);...
            0 curIntrinsic(2) curIntrinsic(4);...
            0 0 1];
        
        X=RT*[cW;zeros(1,size(cW,2));ones(1,size(cW,2))];
        X=X(1:2,:)./repmat(X(3,:),[2 1]);
        if (~isempty(curDist))
            x=curDist; %#ok<NASGU> - used in the eval below
            xl=X-eval(model);
        else
            xl=X;
        end
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
    function [vecX xLower xUpper]=composeOptVector(OT)
        
        vecX=zeros(nConst+5+6*length(usedFrames),1);
        %fx fy cx cy gamma
        vecX(nConst+1:(nConst+5))=[A(1,1) A(2,2) A(1,3) A(2,3) 0];
        for ii=1:length(usedFrames)
            % then R and T for each frame (3 and 3)
            vecX((5+nConst+(6*(ii-1)+1)):(5+nConst+6*ii))=OT(ii,:);
        end
        if (nargout > 1) %we assume the parameter change isnt that big
            maxChange=0.25;
            xLower=vecX;
            smId=abs(vecX)<=1;
            xLower(xLower~=0)=xLower(xLower~=0)-abs(xLower(xLower~=0)*maxChange);
            xLower(smId)=xLower(smId)-1;
            
            xUpper=vecX;
            xUpper(xUpper~=0)=xUpper(xUpper~=0)+abs(xUpper(xUpper~=0)*maxChange);
            xUpper(smId)=xUpper(smId)+1;
            
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

