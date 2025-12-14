%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  File:     inflection.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  definition of inflectional classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% definition of the symbol classes
#include "symbols.fst"

%%% definition of the inflectional classes %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%$ConjReg-icin$ = <conj>:{için}

$NomRegNum-lKrY$ = <sg>:<> | <POSS_pl>:{l<K>r<Y>} 
$NomRegNum-lKr$ = <sg>:<> | <pl>:{l<K>r}
$NomReg-lK$ = <ins>:{l<K>}
$NomReg-ylK$ = <y_ins>:{yl<K>}
$NomReg-mK$ = <CV>:{m<K>}
%$VerbRegDer-lKsch$ = <D_lKsch>:{l<K>ş}
%$VerbRegDer-dYr$ = <D_dYr>:{d<Y>r}
$NomRegNeg-sYz$ = <D_sYz>:{s<Y>z}
$NomReg-lYk$ = <D_lYk>:{l<Y>k}
%$VerbRegDer-Ym$ = <D_Ym>:{<Y>m}
$VerbReg-Yyor$ = <praes>:{<Y>yor}
$VerbReg-YyorsK$ = <impfcondcop>:{<Y>yors<K>}
$VerbReg-aoristsK$ = <aor_cond_cop>:{<aorist>s<K>}
$VerbReg-sK$ = <cond>:{s<K>}
$VerbReg-ysK$ = <cond_cop>:{ys<K>}
$VerbReg-dY$ = <di_past>:{d<Y>}
$VerbReg-ydY$ = <cop_di>:{yd<Y>}
$VerbRegRFL-Yn$ = <rfl>:{<Y>n}
$VerbReg-mYsch$ = <misch_past>:{m<Y>ş}
$VerbReg-ymYsch$ = <cop_misch>:{ym<Y>ş}
$VerbRegF-KcKk$ = <fut>:{<K>c<K>k}
$VerbRegN-KcKk$ = <VN>:{<K>c<K>k}
$VerbReg-Kbil$ = <poss>:{<K>bil}
$VerbRegOptativ-Ky$ = <cop_opt>:{<K>y}
$VerbReg-Kyaz$ = <Kyaz>:{<K>yaz}
$VerbReg-Kdur$ = <Kdur>:{<K>dur}
$VerbReg-Yver$ = <Yver>:{<Y>ver}
$VerbReg-Kgoer$ = <Kgör>:{<K>gör}
$VerbReg-Kgel$ = <Kgel>:{<K>gel}
$VerbReg-Yr$ = <caus_ir>:{<Y>r}
$VerbReg-Kkal$ = <Kkal>:{<K>kal}
$VerbReg-dYr$ = <dir>:{d<Y>r}
$VerbReg-mKlY$ = <obligative>:{m<K>l<Y>}
$VerbRegNeg-mK$ = <neg>:{m<K>}
$VerbReg-mK$ = <converb>:{m<K>}
$VerbReg-mKk$ = <VN>:{m<K>k}
$VerbReg-mKk$ = <converb>:{m<K>k}
$VerbReg-Ysch$ = <recip>:{<Y>ş}
$VerbReg-sch$ = <recip>:{ş}
$VerbReg-K$ = <opt>:{<K>}
$VerbRegPoss-K$ = <poss_voice>:<K>
$VerbRegPass-passiv$ = <passiv>
$VerbReg-aorist$ = <aorist>
$RelReg-ki$ = <rel>:{ki}
$VerbReg-ken$ = <adv_ken>:{ken}
$VerbReg-Yp$ = <adv_Yp>:{<Y>p}
$VerbReg-Kn$ = <adv_Kn>:{<K>n}
$VerbReg-KrKk$ = <adv_KrKk>:{<K>r<K>k}
$NomReg-dYk$ = <converb>:{d<Y>k}
$NomReg-dYk$ = <VN>:{d<Y>k}
$NomReg-ylY$ = <>:{yl<Y>}
$NomReg-lY$ = <>:{l<Y>}
$VerbReg-YncK$ = <converb>:{<Y>nc<K>}
$VerbReg-dYkcK$ = <>:{d<Y>kç<K>}
$VerbReg-icin$ = <conj>:{için}


$NomRegCopCase$ = {<Loc>}:{d<K>} | {<Abl>}:{d<K>n}% | {<Cop-Past>}:{t<Y>}

$NomRegCop$= <PRO>:<> ({<1><sg>}:{<Y>m} |\
                {<2><sg>}:{s<Y>n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{<Y>z} |\
                {<2><pl>}:{s<Y>n<Y>z} |\
                {<3><pl>}:{l<K>r})
$NomRegCop-p$= <PRO>:<> ({<1><sg>}:{m} |\
                {<2><sg>}:{n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{k} |\
                {<2><pl>}:{n<Y>z} |\
                {<3><pl>}:{l<K>r})
$NomReg$ = <POSS>:<> ({<1><sg>}:{m} |\
              {<2><sg>}:{n} |\
              {<3><sg>}:{s<Y>} |\
              {<1><pl>}:{m<Y>z} |\
              {<2><pl>}:{n<Y>z} |\
              {<3><pl>}:{l<K>r<Y>})
$NomRegCase$ = {<Nom>}:{} |\
               {<Gen>}:{<Y>n} |\
               {<Dat>}:{<K>} |\
               {<Acc>}:{<Y>} |\
               {<Loc>}:{d<K>} |\
               {<Abl>}:{d<K>n}
$NomReg-Case$ = {<Nom>}:{} |\
               {<Gen>}:{y<Y>n} |\
               {<Dat>}:{y<K>} |\
               {<Acc>}:{y<Y>} |\
               {<Loc>}:{d<K>} |\
               {<Abl>}:{d<K>n}

$NomRegNum-Sg$ = <sg>:<>
$NomRegNum-Pl$ = <pl>:{l<K>r}
$NomReg$ = ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $NomRegCase$ <>:<F> ({<Loc>}:{d<K>}) <>:<F> $RelReg-ki$) |\
	 ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $NomRegCase$) |\
           ($NomRegNum-lKrY$ (<>:<F> $NomReg$)? <>:<F> ({<Loc>}:{d<K>}) <>:<F> $VerbReg-dYr$) |\
           ($NomRegNum-lKr$ <>:<F> (<>:<F> $NomReg$)? <>:<F> $NomReg-lK$) |\
           (($NomRegNum-lKr$ <>:<F>)? (<>:<F> $NomReg$)? <>:<F> $NomReg-ylK$) |\
           ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $VerbReg-dYr$) |\
           ($NomRegNeg-sYz$ <>:<F> $NomRegCop$) |\
           ($NomRegNum-lKr$ (<>:<F> $NomRegCop-p$)? (<>:<F> $VerbReg-ydY$)?) |\
           ($NomRegNum-Sg$ <>:<F> $NomReg-ylY$) |\
           ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $NomRegCopCase$ <>:<F> $VerbReg-ymYsch$ <>:<F> $NomRegCop$) |\
           ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $NomRegCopCase$ <>:<F> $VerbReg-ydY$ <>:<F> $NomRegCop$) |\
           ($NomRegNum-lKr$ <>:<F> $NomReg$ <>:<F> $NomRegCopCase$ <>:<F> $NomRegCop$)|\
           (($NomRegNeg-sYz$)? (<>:<F> $NomReg$)? <>:<F> $NomRegCase$) |\
           (($NomRegNeg-sYz$)? <>:<F> $NomReg-lYk$ (<>:<F> $NomReg$)? <>:<F> $NomRegCase$) |\
           ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> ({<Loc>}:{d<K>}) <>:<F> $RelReg-ki$)|\
           ($NomRegNum-lKr$ (<>:<F> $NomReg$)? <>:<F> $NomRegCopCase$ <>:<F> $NomRegCop$)

$NomReg-p$ = <POSS>:<> ({<1><sg>}:{<Y>m} |\
             {<2><sg>}:{<Y>n} |\
             {<3><sg>}:{<Y>} |\
             {<1><pl>}:{<Y>m<Y>z} |\
             {<2><pl>}:{<Y>n<Y>z} |\
             {<3><pl>}:{l<K>r<Y>})
$NomReg-p$ = ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomReg-lK$) |\
	   ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomReg-lK$) |\
 	   ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$ <>:<F> $RelReg-ki$ <>:<F> $NomReg-ylK$) |\
	   ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomReg-ylK$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$ <>:<F> $VerbReg-dYr$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $VerbReg-dYr$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomRegCop$)? <>:<F> $VerbReg-dY$) |\
             (($NomReg-p$ <>:<F>)? $NomRegCopCase$ (<>:<F> $VerbReg-ymYsch$)? <>:<F> $NomRegCop$) |\
             ($NomRegNum-Pl$ <>:<F> $VerbReg-sK$) |\
             (($NomRegNum-lKr$ <>:<F>)? $VerbReg-YncK$) |\
             ($NomRegNeg-sYz$ <>:<F> $NomRegCop$) |\
             ($NomRegNum-lKr$ <>:<F> $NomReg-p$ <>:<F> $NomRegCopCase$ <>:<F> $NomRegCop$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$) |\
             ($NomRegNum-lKr$ <>:<F> $NomReg-lYk$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$) |\
             ($NomRegNum-lKr$ <>:<F> $NomReg-p$ <>:<F> $NomReg-lK$) |\
             ($NomRegNum-lKr$ <>:<F> $NomReg-lYk$ <>:<F> $NomReg-p$ <>:<F> $NomReg-lK$) |\
             (($NomRegNeg-sYz$)? (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$) |\ 
             (($NomRegNeg-sYz$)? <>:<F> $NomReg-lYk$ (<>:<F> $NomReg-p$)? <>:<F> $NomRegCase$) |\            
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? <>:<F> ({<Loc>}:{d<K>}) <>:<F> $RelReg-ki$) |\
             ($NomRegNum-lKr$ (<>:<F> $NomReg-p$)? (<>:<F> $NomRegCopCase$)? <>:<F> $NomRegCop$)

$NomReg-su$ = <POSS>:<> ({<1><sg>}:{y<Y>m} |\
             {<2><sg>}:{y<Y>n} |\
             {<3><sg>}:{y<Y>} |\
             {<1><pl>}:{y<Y>m<Y>z} |\
             {<2><pl>}:{y<Y>n<Y>z} |\
             {<3><pl>}:{l<K>r<Y>})

$NomReg-su$ =  ($NomRegNum-Sg$ <>:<F> $NomReg-su$ <>:<F> $NomRegCase$) |\
               ($NomRegNum-Sg$ <>:<F> $NomReg-Case$) |\
               ($NomRegNum-Pl$ (<>:<F> $NomReg-p)? <>:<F> $NomRegCase$)

                
$NomRegCop-p$ = ({<Cop-Past>}:{t<Y>} <>:<F> $NomRegCop-p$)  

$NEReg$ = $NomRegCase$

%%%yor/ir-Praesens%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$VerbReg-Ym$ =  {<1><sg>}:{<Y>m} |\
                {<2><sg>}:{s<Y>n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{<Y>z} |\
                {<2><pl>}:{s<Y>n<Y>z} |\
                {<3><pl>}:{l<K>r}
$VerbReg-Ym$ = (($VerbRegRFL-Yn$ <>:<F>)? $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dYr$ ) |\
			   ($RelReg-ki$)|\
			   ($VerbReg-Yp$)|\
			   ($VerbReg-Kn$)|\
               (($VerbReg-Ysch$ <>:<F>)? $VerbReg-mKk$ <>:<F> {<Loc>}:{d<K>} <>:<F> $VerbReg-Ym$ (<>:<F> $VerbReg-dYr$)?) |\ 
               ($VerbReg-Ysch$ <>:<F> $VerbReg-dYr$ <>:<F> $VerbReg-Kn$) |\ 
               ($VerbReg-Kn$) |\
               ($VerbReg-mKk$) |\               
               (($VerbReg-Ysch$ <>:<F>)? ($VerbRegPass-passiv$ <>:<F>)? <>:<F> {<imp><2><sg>}:{}) |\
               (($VerbRegPass-passiv$ <>:<F>)? <>:<F> {<imp><3><sg>}:{s<Y>n}) |\
               (($VerbRegPass-passiv$ <>:<F>)? <>:<F> {<imp><2><pl>}:{<Y>n}) |\
               (($VerbRegPass-passiv$ <>:<F>)? <>:<F> {<imp><2><pl>}:{<Y>n<Y>z}) |\
               (($VerbRegPass-passiv$ <>:<F>)? <>:<F> {<imp><3><pl>}:{s<Y>nl<K>r}) |\
               (($VerbRegPass-passiv$ <>:<F>)? $VerbRegNeg-mK$ <>:<F> {<imp><2><sg>}:{}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><3><sg>}:{s<Y>n}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><2><pl>}:{<Y>n}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><2><pl>}:{<Y>n<Y>z}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><3><pl>}:{s<Y>nl<K>r}) |\
               ($VerbRegPass-passiv$ (<>:<F> $VerbReg-Yyor$)? <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
               (($VerbRegPass-passiv$ <>:<F>)? $NomReg-dYk$ <>:<F> $NomRegCase$ <>:<F> $NomReg$) |\
               ($NomReg-dYk$ <>:<F> $NomReg-p$ <>:<F> $NomReg-lK$) |\
               (($VerbRegNeg-mK$ <>:<F>)? $NomReg-dYk$ <>:<F> $NomRegCase$ <>:<F> $NomReg-p$) |\
               (($VerbRegNeg-mK$ <>:<F>)? $VerbRegN-KcKk$ <>:<F> $NomRegCase$ <>:<F> $NomReg-p$) |\
               (($VerbRegPass-passiv$ <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)? $VerbRegN-KcKk$ <>:<F> $NomRegCase$ (<>:<F> $NomReg$ <>:<F> $NomRegCase$)?) |\
               (($VerbRegNeg-mK$ <>:<F>)? $NomRegCopCase$) |\
               ($VerbRegNeg-mK$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
               (($VerbRegPass-passiv$ <>:<F>)? $VerbRegNeg-mK$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$ <>:<F> {<Abl>}:{d<K>n}) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-Kbil$ (<>:<F> $VerbReg-Yyor$)? <>:<F>  $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$ (<>:<F> $RelReg-ki$)?) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbRegN-KcKk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-aorist$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-mYsch$  <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dY$) |\
                  ($VerbReg-Yyor$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-Kbil$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-Kbil$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-Kbil$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-aorist$ <>:<F> $VerbReg-Ym$)  |\
                  ($VerbReg-Ysch$ <>:<F> ($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)? $VerbReg-aorist$ (<>:<F> $VerbReg-dY$)? <>:<F> $VerbReg-Ym$) |\
                  (($VerbReg-Ysch$ <>:<F>)? $VerbRegPass-passiv$  <>:<F> ($VerbRegNeg-mK$ <>:<F>)? $VerbReg-aorist$)  |\
                  (($VerbReg-Ysch$ <>:<F>)? $VerbRegPass-passiv$  <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-Ym$)  |\
                  ($VerbReg-aorist$ <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dY$) |\
                  ($VerbRegPass-passiv$  <>:<F> $VerbReg-Ym$)  |\                  
                  ($VerbRegPass-passiv$  <>:<F> $NomReg-Case$)  |\
                  (($VerbRegPass-passiv$  <>:<F>)? $VerbReg-dYkcK$) |\
                  ($VerbRegPass-passiv$  <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-dYr$ ) |\
                  (($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)?  $VerbReg-dYkcK$) |\
                   (($VerbRegPass-passiv$  <>:<F>)? $VerbReg-YncK$) |\
                   (($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)?  $VerbReg-YncK$) |\
                  (($VerbRegPass-passiv$  <>:<F>)?  $VerbReg-KrKk$) |\
                   (($VerbRegPass-passiv$  <>:<F>)?  $VerbReg-Kn$) |\
                  (($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)?  $VerbReg-KrKk$) |\
                  (($VerbRegPass-passiv$  <>:<F>)?  $VerbReg-Kn$ (<>:<F> $NomReg-p$)?) |\
                  (($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)?  $VerbReg-Kn$) |\
                  (($VerbRegPass-passiv$  <>:<F>)? ($VerbRegNeg-mK$ <>:<F>)?  $VerbReg-Yp$) |\
                  (($VerbReg-aorist$  <>:<F>)?  $VerbReg-KrKk$) |\  
                  ($VerbReg-Kkal$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kkal$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kkal$ <>:<F> $VerbReg-mKlY$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kkal$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ymYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kkal$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kdur$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kdur$ <>:<F> $VerbReg-mKlY$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kdur$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ymYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kdur$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kdur$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\ 
                  ($VerbReg-Kgoer$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kgoer$ <>:<F> $VerbReg-mKlY$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgoer$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ymYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgoer$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgoer$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\ 
                  ($VerbReg-Kgel$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kgel$ <>:<F> $VerbReg-mKlY$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgel$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ymYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgel$ <>:<F> $VerbReg-mYsch$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kgel$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kyaz$ <>:<F> $NomReg-dYk$ <>:<F> $NomReg-p$) |\
                  ($VerbReg-Kyaz$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kyaz$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ymYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kyaz$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kyaz$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
                  (($VerbRegNeg-mK$ <>:<F>)? $VerbReg-aorist$ <>:<F>  $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  ($VerbReg-aorist$ (<>:<F> {<3><pl>}:{l<K>r})? <>:<F> $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  ($VerbReg-mYsch$ <>:<F> $VerbReg-ken$ (<>:<F> $RelReg-ki$)? ) |\
                  (($VerbRegNeg-mK$ <>:<F>)? $VerbReg-mYsch$ <>:<F>  $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  ($VerbRegN-KcKk$ <>:<F> $VerbReg-ken$ (<>:<F> $RelReg-ki$)? ) |\
                  (($VerbRegNeg-mK$ <>:<F>)? $VerbRegN-KcKk$ <>:<F>  $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  ($VerbReg-Yyor$ <>:<F> $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  (($VerbRegNeg-mK$ <>:<F>)? $VerbReg-Yyor$ <>:<F>  $VerbReg-ken$ (<>:<F> $RelReg-ki$)?)  |\
                  (($VerbReg-sK$ <>:<F>)? $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-mYsch$ <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dYr$ ) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$ (<>:<F> $VerbReg-dYr$)? )|\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dYr$ ) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbRegOptativ-Ky$ <>:<F> $VerbReg-Ym$ ) |\
                  ($VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dYr$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbRegF-KcKk$ (<>:<F> $VerbReg-mYsch$)? <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-Kbil$ <>:<F> $VerbRegN-KcKk$  <>:<F> $NomRegCase$) |\
                  ($VerbRegF-KcKk$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbReg-mKlY$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-aorist$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-aorist$ ) |\
                  ($VerbRegPass-passiv$  <>:<F> $VerbReg-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-aorist$ ) |\     
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\
                  %($VerbRegPass-passiv$ <>:<F> $VerbRegF-KcKk$  <>:<F> $NomRegCase$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-Yyor$  <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-Ym$ (<>:<F> $VerbReg-dYr$)?) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-Kbil$ <>:<F> $VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-ymYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPass-passiv$ <>:<F> $VerbRegPoss-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbReg-mKk$  <>:<F> $NomRegCopCase$  <>:<F> $VerbReg-Ym$) |\  
                  ($VerbReg-mKk$  <>:<F> ({<Loc>}:{d<K>}) <>:<F> $VerbReg-Ym$ <>:<F> $VerbReg-dYr$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegPoss-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\ 
                  ($VerbRegPoss-K$ <>:<F> $VerbRegNeg-mK$ <>:<F>  $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
                  ($VerbRegOptativ-Ky$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-Ym$)|\
                  ($VerbRegOptativ-Ky$ <>:<F> $VerbReg-Ym$) |\
                  (($VerbReg-Kbil$ <>:<F>)? $VerbRegN-KcKk$ <>:<F> $VerbReg-Ym$ <>:<F> $NomRegCase$) |\
                  ($VerbRegNeg-mK$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-Ym$) 
                  



$VerbRegK$ = {<1><sg>}:{y<Y>m} |\
                {<2><sg>}:{s<Y>n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{l<Y>m} |\
                {<2><pl>}:{s<Y>n<Y>z} |\
                {<3><pl>}:{l<K>r}
$VerbRegK$  = ($VerbReg-K$  <>:<F> $VerbRegK$)|($VerbReg-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-mKk$)
%%%möchte(Optativ)-Praesens%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$VerbRegNeg-Ir$ = {<1><sg>}:{m} |\
                {<2><sg>}:{zs<Y>n} |\
                {<3><sg>}:{z} |\
                {<1><pl>}:{y<Y>z} |\
                {<2><pl>}:{zs<Y>n<Y>z} |\
                {<3><pl>}:{zl<K>r}
$VerbRegNeg-Ir$  = (($VerbReg-K$ <>:<F>)? $VerbRegNeg-mK$  <>:<F> $VerbRegNeg-Ir$) |\
	         ($VerbRegNeg-mK$  <>:<F> $VerbRegNeg-Ir$) 

%%%Konditonal1-(real)wenn-ich-gerade-komme%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$VerbReg-p$ = {<1><sg>}:{m} |\
                {<2><sg>}:{n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{k} |\
                {<2><pl>}:{n<Y>z} |\
                {<3><pl>}:{l<K>r}
$VerbReg-p$ =  ($VerbReg-Yver$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
	     ($VerbRegNeg-mK$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
	     ($VerbReg-Yp$)|\
	     ($VerbReg-Kn$)|\
               (<>:<F> {<imp><2><sg>}:{}) |\
               (<>:<F> {<imp><3><sg>}:{s<Y>n}) |\
               (<>:<F> {<imp><2><pl>}:{<Y>n}) |\
               (<>:<F> {<imp><2><pl>}:{<Y>n<Y>z}) |\
               (<>:<F> {<imp><3><pl>}:{s<Y>nl<K>r}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><2><sg>}:{}) |\
               ($VerbReg-Yr$ <>:<F> $VerbReg-Yyor$ <>:<F> $VerbReg-Ym$) |\
               ($VerbReg-sch$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$)  |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><3><sg>}:{s<Y>n}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><2><pl>}:{<Y>n}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><2><pl>}:{<Y>n<Y>z}) |\
               ($VerbRegNeg-mK$ <>:<F> {<imp><3><pl>}:{s<Y>nl<K>r}) |\
               ($VerbReg-Yver$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-sch$ <>:<F> $VerbReg-dYr$ <>:<F> $VerbReg-Kn$) |\
               ($VerbReg-Kkal$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kkal$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kdur$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kdur$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kyaz$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kyaz$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kgoer$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kgoer$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kgel$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Kgel$ <>:<F> $VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-mKk$  <>:<F> ({<Loc>}:{d<K>}) <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
               (($VerbRegPass-passiv$)? <>:<F> $NomReg-mK$  <>:<F>  $NomReg$) |\
               (($VerbRegPass-passiv$)? <>:<F> $NomReg-mK$  <>:<F>  $NomRegCase$) |\
               (($VerbRegPass-passiv$)? <>:<F> $VerbReg-Kbil$ <>:<F>  $VerbRegN-KcKk$ (<>:<F> $VerbReg-Ym$)? <>:<F> $NomRegCase$) |\
               ($VerbReg-Kbil$ <>:<F>  $VerbRegN-KcKk$ (<>:<F> $NomRegCase$)? <>:<F> $NomReg-p$) |\
               ($VerbReg-aorist$ <>:<F> $VerbReg-Ym$) |\
               (($VerbRegNeg-mK$ <>:<F>)? $VerbReg-sK$ <>:<F> $VerbReg-p$) |\               
               ($VerbReg-mKlY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-mKlY$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\          
               ($VerbReg-mKlY$ <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F>  $NomReg-mK$ <>:<F> $NomReg$ <>:<F> $NomRegCase$) |\
               %($NomReg-mK$ <>:<F> $NomReg$ <>:<F> $NomRegCase$) |\
               ($VerbRegPass-passiv$ <>:<F>  $VerbReg-mKk$ <>:<F> ({<Loc>}:{d<K>})) |\
               %($VerbRegPass-passiv$ <>:<F> $VerbReg-Kbil$ <>:<F> ($NomReg-mK$ <>:<F>)? $VerbReg-aorist$ <>:<F> $NomReg$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-Kbil$ <>:<F> $VerbReg-YyorsK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Kbil$ <>:<F> $VerbReg-YyorsK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Kbil$ <>:<F> $VerbReg-mKk$)|\ 
               ($VerbReg-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-mKk$)|\
               ($VerbReg-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbRegF-KcKk$ <>:<F> $NomRegCase$)|\
               ($VerbReg-K$ <>:<F> $VerbRegNeg-mK$ <>:<F> $VerbReg-mKk$ <>:<F> ({<Loc>}:{d<K>}) <>:<F> $VerbReg-dYr$ <>:<F> 		  $NomRegNum-Pl$)|\
               ($VerbReg-YyorsK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Kbil$ <>:<F> $VerbReg-aoristsK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-aorist$ <>:<F> $VerbReg-dY$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Ysch$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-dY$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-aoristsK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Kbil$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
	     %($VerbRegPass-passiv$ <>:<F> $VerbReg-aorist$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-dY$ <>:<F> $VerbReg-ydY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-dY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbRegNeg-mK$  <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbRegNeg-mK$  <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$ <>:<F> $NomRegCase$) |\
               ($VerbRegNeg-mK$  <>:<F> $VerbReg-aorist$ <>:<F>  $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-aorist$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-sK <>:<F> $VerbReg-p$) |\
               ($VerbReg-dY$ <>:<F> $VerbReg-p$ <>:<F> $VerbReg-sK$ ) |\
               ($VerbReg-Yyor$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-Yyor$ <>:<F> $VerbReg-dY$  <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-mYsch$ <>:<F> $VerbReg-sK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-aorist$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-sK$ <>:<F>  $VerbReg-p$) |\
               ($VerbReg-Yyor$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-p$) |\
               ($VerbReg-mYsch$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegNeg-mK$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegNeg-mK$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegF-KcKk$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-p$) |\
               ($VerbRegF-KcKk$ <>:<F> $VerbReg-Ym$) |\
               ($VerbRegF-KcKk$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegF-KcKk$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> ($VerbReg-aorist$ <>:<F>)?  $VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-aoristsK$ <>:<F> $VerbReg-p$) |\
               ($VerbRegPass-passiv$ <>:<F> $VerbReg-aoristsK$ <>:<F>$VerbReg-dY$ <>:<F> $VerbReg-p$) |\
               (($VerbRegPass-passiv$ <>:<F>)? ($VerbReg-aorist$ <>:<F>)?  $VerbReg-dY$ <>:<F> $VerbReg-ysK$ <>:<F> $VerbReg-p$) |\
               ($VerbRegOptativ-Ky$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$)|\
               ($VerbRegNeg-mK$ <>:<F> $VerbRegOptativ-Ky$ <>:<F> $VerbReg-dY$ <>:<F> $VerbReg-p$ ) |\
              % (($VerbRegPass-passiv$ <>:<F>)? $VerbRegF-KcKk$ <>:<F> $NomRegCase$) |\
               ($VerbRegF-KcKk$ <>:<F> $VerbReg-mYsch$ <>:<F> $VerbReg-sK$ <>:<F> $VerbReg-p$)



$VerbReg$ =  $VerbReg-p$ | $VerbReg-Ym$ | $VerbRegNeg-Ir$ | $VerbRegK$ | $VerbReg-icin$

%%%%%%%%%-ADJEKTIVE-Regeln%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$AdjReg-i$ = {<1><sg>}:{<Y>m} |\
                {<2><sg>}:{s<Y>n} |\
                {<3><sg>}:{} |\
                {<1><pl>}:{<Y>z} |\
                {<2><pl>}:{s<Y>n<Y>z} |\
                {<3><pl>}:{l<K>r}
$AdjReg-i$ = (<>:<F>) | $NomReg-p$ | $NomReg$ | ($AdjReg-i$) | ($AdjReg-i$ <>:<F> $NomRegCop$) | ( ($AdjReg-i$ <>:<F>)? $VerbReg-dYr$ ($AdjReg-i$ <>:<F>)?)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            CONJUNCTİONS            %
%$ConjReg$ = $ConjReg-icin$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% adding a tag for the inflectional class
$LCInfl$ = <>:<NomReg>  $NomReg$ |\
           <>:<NomReg-p> $NomReg-p$  |\
           <>:<NomRegCop> $NomRegCop$ |\
           <>:<NomReg-su> $NomReg-su$  |\
           <>:<VerbReg> $VerbReg$ |\
           %<>:<ConjReg>  $ConjReg$ |\
           <>:<AdjReg-i> $AdjReg-i$ 
%           <>:<AdjReg-i-comp> $AdjReg-i-comp$
% no capitalized or fixed word forms yet
% $UCInfl$ = ...
% $FixInfl$ = ...

% The capitalization of the resulting word form is indicated by
% the three feature tags <LC> (lower case the first character),
% <Cap> (capitalize the first character) and <Fix> (do nothing)

$LCInfl$ = $LCInfl$ <>:<LC>
$UCInfl$ = <>:<NEReg> $NEReg$
 $UCInfl$ = $UCInfl$ <>:<UC>
% $FixInfl$ = $FixInfl$ <>:<Fix>

$LCInfl$  | $UCInfl$ %| $FixInfl$

