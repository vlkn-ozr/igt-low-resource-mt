%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  File:     compoundfilter.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  enforcement of compounding constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#include "symbols.fst"


% Compounding filter
% Compounds are restricted to nouns and adjectives

$org$ = [#Origin#]:<>

% symbols occurring in non-compounds
$T$ = [#Letter# #EntryType#] | [#WordClass#]:<> | $org$

% expression matching non-compounds
$TS$ = $T$*

% expression matching compounds
$TC$ = ($T$ | <comp>:<>)*


($TS$ [<NE><ADV><CARD><V><OTHER>] |\
 $TC$ [<ADJ><N>]) \
<base>:<> $org$ [#InflClass#]
