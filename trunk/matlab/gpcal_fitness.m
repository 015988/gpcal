function [fitness gp]=gpcal_fitness(evalstr,gp)

model='d0';

for curGene=1:length(evalstr)
    model=sprintf('%s+d%1g*(%s)',model,curGene,evalstr{curGene});
end


model=strrep(model,'x8','u(X)');
model=strrep(model,'x9','v(X)');


if (isempty(strfind(model,'u(X)'))||isempty(strfind(model,'v(X)')))
    fitness=inf;
else
    nCases=size(gp.userdata.X,2);
    X=gp.userdata.X;
    x=gp.userdata.x;
    used=gp.userdata.F;
    As=gp.userdata.A;
    OTs=gp.userdata.OT;
    t_fit=NaN(nCases,1);
    %%%%
    parfor c=1:nCases
        t_fit(c)=ComputeCalibration(X(c).X,x(c).x,model,used(c).usedFrames,As(c).A,OTs(c).OT);
    end
    gp.userdata.F=used;
    fitness=mean(t_fit);
    %if ((~isnan(fitness))&&(~isinf(fitness)))
        f=fopen('log.txt','a');
        fprintf(f,'%g -- ',fitness);
        fprintf(f,'%g ',t_fit);
        fprintf(f,' -- %s \n',model);
        fclose(f);
%    end
    
end
end