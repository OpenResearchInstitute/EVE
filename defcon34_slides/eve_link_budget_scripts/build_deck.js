const pptxgen = require("pptxgenjs");
const path = require("path");
const FIG = path.resolve(__dirname, "../figs"); const p = f => path.join(FIG, f);
const C = { bg:"0A0E0A", panel:"0F1610", green:"39FF14", green2:"1FBF0A", cyan:"00E5FF",
  amber:"FFB000", magenta:"FF5CF4", text:"C8FFC8", dim:"5F8F66" };
const MONO = "Courier New";
const pres = new pptxgen(); pres.defineLayout({name:"W",width:13.3,height:7.5}); pres.layout="W";
const bg = s => s.background = { color: C.bg };
function tag(s,t){ s.addText(t,{x:0.45,y:0.22,w:12.4,h:0.4,fontFace:MONO,fontSize:15,color:C.green,align:"left",margin:0}); }
function figSlide(f,t,notes){ const s=pres.addSlide(); bg(s); tag(s,t);
  s.addImage({path:p(f),x:1.05,y:0.82,w:11.2,h:6.3}); s.addNotes(notes); return s; }

// 1 title
const t=pres.addSlide(); bg(t);
t.addText("HACK THE PLANET",{x:0.5,y:2.05,w:12.3,h:1.2,fontFace:MONO,fontSize:58,bold:true,color:C.green,align:"center",charSpacing:2});
t.addText("( not ours )",{x:0.5,y:3.25,w:12.3,h:0.6,fontFace:MONO,fontSize:24,color:C.dim,align:"center"});
t.addText("E A R T H   ->   V E N U S   ->   E A R T H",{x:0.5,y:4.05,w:12.3,h:0.5,fontFace:MONO,fontSize:20,color:C.cyan,align:"center"});
t.addText([{text:"Michelle Thompson  W5NYV",options:{breakLine:true,color:C.text}},{text:"Open Research Institute",options:{breakLine:true,color:C.dim}}],{x:0.5,y:4.9,w:12.3,h:1.0,fontFace:MONO,fontSize:18,align:"center"});
t.addText("RF VILLAGE // DEF CON   >_",{x:0.5,y:6.55,w:12.3,h:0.5,fontFace:MONO,fontSize:16,color:C.amber,align:"center"});
t.addNotes("Hi, I'm Michelle Thompson, W5NYV, with Open Research Institute. This talk is called Hack the Planet, and we are going to bounce a signal off Venus and hear it back on Earth. First I'll get our signal there and back, and then I'll show you the waveform, designed by my colleague Pete Wyckoff, that actually makes it work. Let's break in.");

figSlide("fig_bochum.png","> 02 // 2009 // the dead carrier",
"It starts in 2009. On March 25th, a team from AMSAT-DL aimed a dangerously powerful magnetron transmitter at the Bochum observatory at Venus. About five minutes later, the round-trip light time, they heard their own echo come back off the surface of another planet. First time amateurs ever did that, a genuine landmark. But look at what came back: a single line. A dead carrier. A pure tone with no information in it. Detecting a tone is one thing. We wanted to do something much harder.");
figSlide("fig_carrier_vs_comms.png","> 03 // carrier vs communications",
"Here is the whole problem in one picture. On the left, 2009: a dead carrier; you just have to detect a line. On the right, what we want: a communications signal, actual bits. That raises the bar twice. Information needs more signal-to-noise than a bare carrier, and Venus spins, so the echo comes back smeared across frequency, Doppler spread. A stationary mirror does not do that; a spinning planet does. We need more margin AND a signal that survives being smeared. That is the hack.");
figSlide("fig_crew.png","> 04 // the crew + the timing",
"Who, and when? The when is brutal: Earth and Venus line up only about every eighteen months, at inferior conjunction. Unlike the Moon, Venus is barely close enough even at its closest for the largest amateur dishes on Earth. The who: our crew is Dwingeloo, the 25-meter dish in the Netherlands, at 1299.5 megahertz. And our phone-a-friend is Effelsberg, the 100-meter giant in Germany. Timing and coordination here are every bit as hard as the radio.");
figSlide("fig_bridge.png","> 05 // the budget // which part matters",
"The link budget: every gain and loss from transmitter to far end. A thousand watts and dish gain, then the cliff, a round-trip path loss near 494 dB. Venus gives some back, the albedo takes a chunk, and with Dwingeloo's measured 65-kelvin system temperature we land, for Dwingeloo alone in October, at minus 1.33 dB-Hz. Below zero. Which part matters most? Path loss dominates, and you fix that with aperture. Hold that thought.");
figSlide("fig_bounce.png","> 06 // the bounce // ~13% back",
"Venus as a mirror: huge, but lousy. Only about 13 percent of the energy that hits it comes back; the rest scatters. Depending on date and band that is a radar albedo between roughly 12 and 15 percent, and for October we use the date-specific dynamic value. Because it is rough and spinning, what does come back is smeared. A huge, lousy, spinning mirror.");
figSlide("fig_why_1hz.png","> 07 // why 1 Hz",
"Why do we quote everything in one hertz? People think it means a one-hertz radio, which sounds like cheating. It is not. One hertz is a ruler, not a receiver. Signal-to-noise in any bandwidth is the carrier-to-noise density minus ten log of the bandwidth, one line, every point the same signal off Venus. Quoting per hertz states the noise as a density, so the number describes the planet bounce. And it is literally how we measured the real echoes.");
figSlide("fig_validation.png","> 08 // we already measured it // CAMRAS 2025",
"Here is why you should believe these numbers. In March 2025, CAMRAS at Dwingeloo bounced carriers off Venus, only the second amateur EVE detection ever. We measured the carrier-to-noise density in one hertz from all four echoes: mean plus 0.645 dB. Our link budget predicted plus 0.560. A residual of eighty-five thousandths of a dB. We did not model Venus and hope. We measured it, and the model was right to a tenth of a dB.");
figSlide("fig_rescue.png","> 09 // October 2026 // the rescue",
"October, on that validated model, using the harder dynamic albedo. Dwingeloo alone peaks October 25th at minus 1.33 dB-Hz. Below zero. Then Effelsberg picks up the phone: the 100-meter dish plus a colder receiver adds 13.27 dB, rock solid across the whole window, independent of how bright Venus is that day. That lifts us to plus 11.94 dB-Hz. Alone we sink; together we get in. Effelsberg de-risks the whole attempt.");

figSlide("fig_handoff.png","> 10 // the spin damages information",
"But even with Effelsberg's margin, there is a catch. Venus spins, so the echo of a clean tone comes back Doppler-spread, about plus or minus three-quarters of a hertz measured by CAMRAS, and drifting. Every off-the-shelf weak-signal mode assumes a stable tone for hundreds of seconds; Venus refuses. So no existing amateur protocol closes this link. The obvious question is: what does? Let me show you exactly what the spin breaks, and the waveform we built to beat it.");
figSlide("fig_coherence.png","> 11 // doppler spread -> coherence time",
"Here is the real damage, not just that it is smeared. The spread sets a coherence time, the reciprocal of the spread, about a third of a second for our October forecast. That is the longest you can integrate coherently before Venus scrambles the phase. The dashed line is the sensitivity you would get if you could integrate coherently forever; the solid line is what you actually get, coherent up to the coherence time, then non-coherent combining, which only buys half the dB. That red gap is the coherence-time tax: over 13 dB at a 165-second symbol. That tax dictates the whole waveform.");
figSlide("fig_zadoff.png","> 12 // candidate: Zadoff-Chu",
"One natural candidate is a Zadoff-Chu sequence, a constant-amplitude chirp, the spiral. Two lovely properties: constant envelope, so it loves a high-power transmitter; and a razor-sharp autocorrelation, one spike, superb for finding the echo and nailing its delay and Doppler. The chirp even tolerates the big Doppler shift, because a shift just slides the peak. So Zadoff-Chu is strong for acquisition and as an alphabet of orthogonal symbols. But its coherent correlation still cannot exceed the coherence time. It does not escape the tax; it pays it gracefully.");
figSlide("fig_pete_design.png","> 13 // Pete's Spiral // the design that fits",
"Here is the design Pete Wyckoff built, and it is beautiful because every number is chosen around that coherence time. 106 bits, BCH-coded, mapped to 4096-ary orthogonal symbols, generated as spirals. The key: the FFT bin width is 2.87 hertz, exactly the Doppler-spread forecast, so each coherent frame is about a third of a second, right at the coherence time, and the tones are spaced wider than the spread so they never smear together. Each 165-second symbol is 440 such frames combined non-coherently, building sensitivity without ever integrating coherently past the coherence time. A waveform shaped entirely by the spinning planet.");
figSlide("fig_mary.png","> 14 // why M-ary orthogonal",
"And why 4096-ary? Because this link is power-limited, not bandwidth-limited. Orthogonal signaling gets more power-efficient as the alphabet grows: binary to 4096-ary buys about seven dB, landing within a few dB of the ultimate limit, minus 1.59 dB. We spend 22 kilohertz of bandwidth, which is free out here, to save precious dBs of power. Exactly the right trade for a planet bounce.");
figSlide("fig_eme.png","> 15 // prove it on the Moon // EME testbed",
"How do we test all this before the one shot in October? The Moon. Earth-Moon-Earth is the perfect testbed. The Moon's libration gives an even harsher Doppler spread than Venus, six hertz versus our three, measured by Joe Taylor at 1296 megahertz. But the Moon is 223 dB closer in path loss, so the echo is enormously strong. We stress-test the exact coherence-time strategy, harder than Venus will, but with signal to spare, using the same dishes, the same band, the same pipeline. And the Moon is up most nights. Prove it on the Moon, then take the shot at Venus.");

// 16 close
const c=pres.addSlide(); bg(c);
c.addText("LET'S HACK A PLANET",{x:0.5,y:1.6,w:12.3,h:1.0,fontFace:MONO,fontSize:46,bold:true,color:C.green,align:"center",charSpacing:1});
c.addText([
 {text:"the spin damages information in a measurable way", options:{breakLine:true,color:C.text}},
 {text:"we shaped a waveform around its coherence time", options:{breakLine:true,color:C.text}},
 {text:"Effelsberg gives us the margin", options:{breakLine:true,color:C.text}},
 {text:"the Moon lets us prove it first  ->  Venus, October 2026", options:{breakLine:true,color:C.cyan}},
],{x:0.5,y:3.0,w:12.3,h:2.0,fontFace:MONO,fontSize:18,align:"center",lineSpacingMultiple:1.3});
c.addText("waveform: Pete Wyckoff KA3WCA  //  CAMRAS Dwingeloo  //  Effelsberg  //  ORI",{x:0.5,y:5.6,w:12.3,h:0.5,fontFace:MONO,fontSize:14,color:C.dim,align:"center"});
c.addText("openresearch.institute   >_",{x:0.5,y:6.5,w:12.3,h:0.5,fontFace:MONO,fontSize:16,color:C.amber,align:"center"});
c.addNotes("So that is the hack. A spinning planet damages information in a specific, measurable way; we shaped a waveform around its coherence time; Effelsberg gives us the margin; and the Moon lets us prove it all first. In October, Earth and Venus line up again, and we will be pointed at it. Thanks to Pete Wyckoff for the waveform, to CAMRAS and Dwingeloo and Effelsberg, and to Open Research Institute. Let's hack a planet.");

pres.writeFile({fileName:path.resolve(__dirname,"../HackThePlanet_EVE.pptx")}).then(f=>console.log("wrote",f));
