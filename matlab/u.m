function out=u(in)
out=in;
if (all(size(in)>1))
    out(2,:)=0;
else
    out=NaN;
end

end