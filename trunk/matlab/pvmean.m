function out=pvmean(in)
if (any(size(in)>1))
    t=mean(in');
    out=t(2);
else
    out=NaN;
end
end