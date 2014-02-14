clear
dat=load('4.dat');
I=zeros(2*size(dat,1),size(dat,2)/2);
for i=1:size(dat,1)
    I(2*(i-1)+1,:)=dat(i,1:2:end);
    I(2*(i-1)+2,:)=dat(i,2:2:end);
end

ref=load('4.ref');
W=zeros(2*size(ref,1),size(ref,2)/3);
for i=1:size(ref,1)
    W(2*(i-1)+1,:)=ref(i,1:3:end);
    W(2*(i-1)+2,:)=ref(i,2:3:end);
end

%I=I(1:2,:);
%W=W(1:2,:);
clear dat i ref


% nFrames=size(In,1)/2;
% f=fopen('in.ref','w');
% for i=1:nFrames
%     for p=1:size(Wn,2)
%         fprintf(f,'%g %g ',In(2*(i-1)+1,p),In(2*i,p));
%     end
% end
% fclose(f);    
%     
%     
    
    
    
    
    
    