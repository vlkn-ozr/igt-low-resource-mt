%*************************************************************************
%  File:     phon.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  morphophonological rules
%**************************************************************************
#include "symbols.fst"

ALPHABET = [#EntryType# #Letter# #WordClass# #Cap#] <F>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$R0$ = ((ye:i|de:i)<V><>:y) ^-> (<BaseStem>__<K>)                         
$R0a$ = (<>:y(<Y>ver|<Y>p|<K>r<K>k|<K>bil|<K>c<K>|<K><F>|<Y>nc<K>|<K>n)) ^-> ([#vowel#][<V>]__)               
$R0b$ = ((l<Y>|d<K>|l<K>|m<K>)<F><>:y) ^-> (__(<Y>|<K>))            
$R0c$ = (<>:y(l<K><F>)) ^-> ((s<Y>|l<K>r<Y>|[#vowel#]<N>)__)                           
$R0d$ = (<>:y(l<K><F>)) ^-> ([#vowel#]<N><F>?__)                    
$R0e$ = (<>:y(<K>y)) ^-> (<V>(m<K>)<F>__)                            
$R0f$ = ({<del>}:{<DEL>} ^-> (__[#vowel#][#cons#][#WordClass#]<passiv>)) || (<del>:<> ^-> ()) || (<DEL>:<del> ^-> ())  
$R0g$ = <passiv>:n ^-> ([#vowel#]<V> __)
$R0h$ = <passiv>:{<Y>n} ^-> (l<V> __)
$R0_h$ = <passiv>:{<Y>l} ^-> (r<V> __)
$R00h$ = <passiv>:{<Y>n} ^-> (<V><aorist><F>? __)
$R000h$ = <passiv>:{<Y>l} ^-> (<V><aorist><F>? __)
$R0i$ = <passiv>:{<Y>l} ^-> ((<V>|d<Y>r)<F>? __)
$R0j$ = <aorist>:{<Y>r} ^-> ([#vowel#][#cons#]+[#vowel#][#cons#]+[<V><F>] __) 
$R0k$ = <aorist>:{<Y>r} ^-> (<V>(n|<Y>n|<Y>l)<F> __)
$R0l$ = <aorist>:{<Y>r} ^-> (<BaseStem>(al|ol|öl|gel|kal|bul|bil|var|ver|vur|gör|san|dur)<V> __)
$R0m$ = <aorist>:{r} ^-> ([#vowel#][<V><F>] __)
$R0n$ = <aorist>:{<K>r} ^-> (<V> __)                                         
$R0o$ = r:z ^-> (m<K> <F>__)                                                 

$R0p$ = ([#WordClass#] <F> <>:n (d|<Gen>|l<K>r)) ^-> () 	  		
$R0q$ = {l<K>r}:{} ^-> (__<F> (l<K>r<Y>))
$R0r$ = {s<K> <F>}:{} ^-> (<N>[#letter#]*__(l<K>r))
$R0s$ = (<>:n<Y>n) ^-> ([#vowel#]<N><F>? __)                       
$R0t$ = (<>:y['<K><Y>']) ^-> ([#vowel#][<Suffix><NE><N><ADJ>]<F>? __)            
$R0u$ = (<>:n(<Y>|<K>)) ^-> ([#vowel#](<N><F>?)__)                            
$R0v$ = (<>:n<F>(<Y>|<K>|d<K>)) ^-> (<N><F>?(s<Y>|n<K>|n<Y>)__)                
$R0y$ = (s:<><Y>) ^-> (l<K>r<F>__)
$R0z$ = (<>:<Y><F>) ^-> (l<K>r <F>?__[mn])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%KONSONANTENHARMONIE%%bzw.%%AUSLAUTVERHÄRTUNG%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$R1$ = (k:ğ) ^-> ( <K>c<K>__ <F>? <Y>)                               
$R1a$ = (k:ğ) ^-> (__[<compound>]<F>?)                                     
$R1b$ = (k:ğ) ^-> ((l<Y>|d<Y>|l<K>)__<F>?[#vowel#])
$R1c$ = (b:p|c:ç|d:t|g:ğ) ^-> (__[<compound><N><ADJ>] <F>? [#cons# <LC>] )   

$R1d$ = (d:t|c:ç) ^-> ([fhsşptkç]<X>?[<V><N><ADJ><NE><Suffix><F>] <F>?__)       
$R1e$ = t:d ^-> ([vylmnrz #vowel#] [<V><N><ADJ><Suffix> <F>]<F>? __)           
#=C# = #cons# %#letter#                         
$R1f$ = ([#=C#]<X>:[#=C#]) ^-> (__[#WordClass#]<F>?[#vowel#])                                                                         
$R1g$ = <X>:<> ^-> ()
$R1h$ =[#vowel#]:<> ^->([#vowel#<F>#WordClass#<Suffix>][#cons#]*__ [#WordClass#<Suffix> <F>][#vowel#][#cons#])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%VOKALHARMONIE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
$R2a$ = ([<Y>]:ı ^-> ([aı][<Q><del>]? [<Suffix>#cons##WordClass#<F>]*__))
$R2b$ = ([<Y>]:u ^-> ([ou] [<Q><del>]? [<Suffix> #cons##WordClass#<F>]*__))
$R2c$ = ([<Y>]:ü ^-> ([öüôû] [<Q><del>]? [<Suffix>#cons##WordClass#<F>]*__))
$R2d$ = ([<Y>]:i ^-> ([ieâ] [<Q><del>]? [<Suffix>#cons##WordClass#<F>]*__))
$R2e$ = ([<K>]:e ^-> ([eiöüâûô] [<Q><del>]? [<Suffix>#cons##WordClass#<F>]*__))     
$R2f$ = ([<K>]:a ^-> ([aouı] [<Q><del>]? [<Suffix>#cons##WordClass#<F>]*__))  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
$R3a$ = {<Q>[#vowel#]}:{} ^-> (__[#cons#][#WordClass#<F>]+[#vowel#] ) || (<Q>:<> ^-> ())          
$R3b$ = {<Q>[#vowel#]}:{} ^-> ()
$R3c$ = [#vowel#]:<> ^-> (__[<V><N><ADJ><Suffix> <F>][#vowel#] )
$R3d$ = {<del>[#vowel#]}:{} ^-> ()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$R4$ = {<NE>}:{'<NE>} ^-> (__ [#letter#])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ALPHABET = [#Letter#] [#WordClass#]:<> <F>:<>

$T$ = .* \
	([#EntryType#]:<> ([#LETTER#]:[#letter#] | [#letter#]) .*)* \
	[#Cap#]

$Capitalisation$ = $T$ || (\
	([#LETTER#]:[#letter#] | [#letter#]) .* <LC>:<> |\
	([#letter#]:[#LETTER#] | [#LETTER#]) .* <UC>:<> |\
	.* <FC>:<> )

$X0$ = $R0$||$R0a$||$R0b$||$R0c$||$R0d$||$R0e$||$R0f$||$R0g$||$R0h$||$R0_h$||$R00h$||$R000h$||$R0i$||$R0j$||$R0k$||$R0l$||$R0m$||$R0n$||$R0o$|| $R0p$||$R0q$||$R0r$||$R0s$||$R0t$||$R0u$||$R0v$||$R0y$||$R0z$
$X1$ = $R1$||$R1a$||$R1b$||$R1c$||$R1d$||$R1e$||$R1f$||$R1g$||$R1h$
$X2$ = $R2a$||$R2b$||$R2c$||$R2d$||$R2e$||$R2f$
$X3$ = $R3a$||$R3b$||$R3c$||$R3d$||$R4$||$Capitalisation$


$X$ = $X2$ || $X2$ || $X2$ || $X2$ || $X2$

$PHON1$ = $X0$ ||$X1$
$PHON2$ = $X$ || $X3$
