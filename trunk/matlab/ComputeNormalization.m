function [nP T]=ComputeNormalization(points)
%function [nP T]=ComputeNormalization(points)
% normalizes the data so it is centered in zero and not that spread around
% See "In defence of the eight point algorithm" from Richard Hartley to
% know why. (And Revisiting the eight ... by Chojnacki et al - PAMI03)
% nP is the normalized version of the  points, T is the transformation
% such that nP=T*points
% ONLY FOR BIDIMENSIONAL DATA
% Ignores -1 on data

m=points(sum(points==-1,2)==0,:); %shorthand + ignoring -1s


mbar=ComputeCentroid(points,2);


tV1=m(1:2:end,:);
mt(1,:)=reshape(tV1,[1 numel(tV1)])-mbar(1);
tV2=m(2:2:end,:);
mt(2,:)=reshape(tV2,[1 numel(tV2)])-mbar(2);

modMT=sum(mt.^2,1);
s=sqrt(sum(modMT)/(2*size(mt,2)));

T=[1/s  0  -mbar(1)/s;
    0  1/s -mbar(2)/s;
    0   0      1     ];
nFrames=size(points,1)/2;
nP=-ones(size(points));

for i=1:nFrames
    pT=points( (2*(i-1)+1):(2*i) , :);
    if (~any(pT(1,:)==-1))
        pT=T*[pT; ones(1,size(points,2))];
        pT=pT(1:2,:)./repmat(pT(3,:),[2 1]);
        nP( (2*(i-1)+1):(2*i),:)=pT;
    end
end


end