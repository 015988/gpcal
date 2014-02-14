function centroid=ComputeCentroid(points,nDim)
%function centroid=ComputeCentroid(points,nDim)
% Compute the centroids from points.
% size(points)=(nDim*nImages,nPointsPerImage)
% Ex.: 6 points per image, 10 images, bidimensional points -> (2*10,6)
% size(centroid)=(nDim,1);
for i=1:nDim
    tV=points(i:2:end,:);
    pp(i,:)=reshape(tV,[1 numel(tV)]);
end

centroid=sum(pp,2)/numel(tV);