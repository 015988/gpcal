function r=swap(uv)
%Swaps u for v
if all(size(uv)>1)
    r=[uv(2,:); uv(1,:)];
else
    r=NaN;
end
end