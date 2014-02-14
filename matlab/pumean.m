function out=pumean(in)
if (any(size(in)>1))
    t=mean(in');
    out=t(1);
else
    out=NaN;
end
end