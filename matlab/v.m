function out=v(in)
out=in;
if (all(size(in)>1))
    out(1,:)=0;
else
    out=NaN;
end
end