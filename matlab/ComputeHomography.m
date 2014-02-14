function H=ComputeHomography(W,I)
%function H=ComputeHomography(W,I)
% size(W)=(2,n), size(I)=(2,n)

W(3,:)=1;
I(3,:)=1;
Z=[0 0 0];
A=zeros(3*size(W,2),9);
for p=1:size(W,2)
    A((3*(p-1)+1),:)=[Z -W(:,p)' I(2,p)*W(:,p)' ];
    A((3*(p-1)+2),:)=[W(:,p)' Z -I(1,p)*W(:,p)' ];
    A((3*(p-1)+3),:)=[-I(2,p)*W(:,p)' I(1,p)*W(:,p)' Z ];
end

[~,~,V] = svd(A,'econ');
H = V(:,size(A,2));
H=reshape(H,[3 3])';
H = H./H(3,3);

end