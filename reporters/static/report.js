// report.js - iteration toggling and Chart.js rendering
(function(){
    function applyFilter(val){
        document.querySelectorAll('[data-iteration]').forEach(el => {
            const attr = el.getAttribute('data-iteration');
            if(!attr) return;
            if(val === 'all'){
                el.classList.remove('hidden');
            } else {
                if(attr === String(val)){
                    el.classList.remove('hidden');
                } else {
                    el.classList.add('hidden');
                }
            }
        });
    }

    function renderCharts(){
        try{
            console.debug && console.debug('renderCharts start', {hasCharts: !!window.__REPORT_CHARTS__, ChartDefined: typeof Chart !== 'undefined'});
        }catch(e){}
        if(!window.__REPORT_CHARTS__) return;
        Object.keys(window.__REPORT_CHARTS__).forEach(key => {
            try{
                let data = window.__REPORT_CHARTS__[key] || {};
                const labels = data.labels || [];

                function destroyIfExists(c){
                    if(!c) return;
                    try{
                        // prefer Chart.js utility to find chart instance
                        if(typeof Chart !== 'undefined' && typeof Chart.getChart === 'function'){
                            const existing = Chart.getChart(c);
                            if(existing && typeof existing.destroy === 'function') existing.destroy();
                        }
                        if(c._chartInstance && typeof c._chartInstance.destroy === 'function'){
                            c._chartInstance.destroy();
                        }
                    }catch(e){ console.warn('destroy chart error', e); }
                    try{ delete c._chartInstance; }catch(e){}
                }
                function createChart(canvasId, datasetLabel, dataArr, options, prefer){
                    const canvas = document.getElementById(canvasId);
                    if(!canvas) return;
                    destroyIfExists(canvas);
                    const ctx = canvas.getContext('2d');

                    // allow prefer to be provided either as 5th arg or via options.prefer
                    const preferMode = (typeof prefer !== 'undefined' && prefer !== null) ? prefer : (options && options.prefer);
                    // decide chart type: radar for >=3 dims, else bar
                    const chartType = (labels && labels.length >= 3 && preferMode !== 'bar') ? 'radar' : 'bar';

                    if(chartType === 'radar'){
                        const cfg = {
                            type: 'radar',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: datasetLabel,
                                    data: dataArr || [],
                                    backgroundColor: 'rgba(99,102,241,0.18)',
                                    borderColor: 'rgba(99,102,241,1)',
                                    borderWidth: 2,
                                    pointBackgroundColor: 'rgba(16,185,129,1)',
                                    pointBorderColor: '#fff',
                                    pointRadius: 4,
                                    pointHoverRadius: 6,
                                    pointHoverBackgroundColor: '#fff',
                                    pointHoverBorderColor: 'rgba(16,185,129,1)',
                                    fill: true
                                }]
                            },
                            options: Object.assign({
                                responsive: true,
                                maintainAspectRatio: false,
                                elements: { line: { borderWidth: 2 } },
                                scales: { r: { beginAtZero: true, suggestedMax: 1, ticks: { stepSize: 0.25 } } },
                                plugins: defaultPluginsOptions()
                            }, options || {})
                        };
                        try{ if(typeof Chart !== 'undefined') canvas._chartInstance = new Chart(canvas, cfg); }catch(e){ console.error('chart create error', e, cfg); }
                        return;
                    }

                    // bar chart fallback for 1-2 dims
                    const cfgBar = {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: datasetLabel,
                                data: dataArr || [],
                                backgroundColor: 'rgba(99,102,241,0.6)',
                                borderColor: 'rgba(99,102,241,1)',
                                borderWidth: 1,
                                barPercentage: 0.8,
                                categoryPercentage: 0.85,
                                fill: true
                            }]
                        },
                        options: Object.assign({
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: { ticks: { color: '#374151' } },
                                y: { beginAtZero: true }
                            },
                            plugins: defaultPluginsOptions()
                        }, options || {})
                    };
                    try{ if(typeof Chart !== 'undefined') canvas._chartInstance = new Chart(canvas, cfgBar); }catch(e){ console.error('chart create error', e, cfgBar); }
                }

                function buildDataFromTable(iterKey){
                    const container = document.querySelector('[data-iteration="' + iterKey + '"]');
                    if(!container) return null;
                    // find the tasks table within this iteration block
                    const rows = container.querySelectorAll('table tbody tr');
                    const agg = {};
                    const order = ['T1','T2','T3','T4'];
                    rows.forEach(r => {
                        try{
                            const cells = r.querySelectorAll('td');
                            if(!cells || cells.length < 5) return;
                            const level = (cells[2].textContent || '').trim();
                            const execTimeText = (cells[3].textContent || '').trim().replace('s','');
                            const scoreText = (cells[4].textContent || '').trim();
                            const exec = parseFloat(execTimeText) || 0;
                            const score = parseFloat(scoreText) || 0;
                            if(!level) return;
                            if(!agg[level]) agg[level] = {count:0, sumScore:0, passed:0, sumTime:0};
                            agg[level].count += 1;
                            agg[level].sumScore += score;
                            agg[level].sumTime += exec;
                            // determine pass: score >= 0.5 or PASS badge
                            const passBadge = r.querySelector('td:nth-child(2) .inline-flex');
                            let isPass = false;
                            if(passBadge && /PASS/i.test(passBadge.textContent)) isPass = true;
                            if(score >= 0.5) isPass = true;
                            if(isPass) agg[level].passed += 1;
                        }catch(e){ }
                    });
                    // build arrays in order: prefer T1..T4 then any other levels
                    const labels = [];
                    const scorer = [];
                    const tsr = [];
                    const latency = [];
                    // include known order first
                    order.forEach(lv => { if(agg[lv]) labels.push(lv); });
                    // include any other levels discovered
                    Object.keys(agg).forEach(lv => { if(labels.indexOf(lv) === -1) labels.push(lv); });
                    labels.forEach(lv => {
                        const s = agg[lv];
                        if(!s || s.count === 0){ scorer.push(0); tsr.push(0); latency.push(0); }
                        else{
                            scorer.push(s.sumScore / s.count);
                            tsr.push(s.passed / s.count);
                            latency.push(s.sumTime / s.count);
                        }
                    });
                    return { labels: labels, scorer: scorer, tsr: tsr, latency: latency };
                }

                function defaultPluginsOptions(){
                    return {
                        legend: { labels: { color: '#374151' } },
                        tooltip: {
                            enabled: true,
                            callbacks: {
                                title: function(items){
                                    if(!items || !items.length) return '';
                                    return items[0].label || '';
                                },
                                label: function(context){
                                    var lab = context.dataset && context.dataset.label ? context.dataset.label : '';
                                    var raw = context.raw !== undefined ? context.raw : context.parsed;
                                    var num = Number(raw) || 0;
                                    if(!lab) return String(num);
                                    var ll = lab.toLowerCase();
                                    if(ll.indexOf('latency') !== -1){
                                        return lab + ': ' + num.toFixed(2) + ' s';
                                    }
                                    if(ll.indexOf('tsr') !== -1 || ll.indexOf('success') !== -1){
                                        return lab + ': ' + (num * 100).toFixed(1) + ' %';
                                    }
                                    if(ll.indexOf('score') !== -1){
                                        return lab + ': ' + (num * 100).toFixed(1) + ' %';
                                    }
                                    return lab + ': ' + num.toFixed(2);
                                }
                            }
                        }
                    };
                }

                // if incoming data is empty, try to derive from table DOM as a fallback
                if((!data || !data.labels || !data.labels.length) && typeof buildDataFromTable === 'function'){
                    try{ const derived = buildDataFromTable(key); if(derived) data = derived; }catch(e){ console.warn('derive chart data error', e); }
                }

                // scorer: values expected between 0..1
                createChart('chart-iter-' + key + '-scorer', 'Overall Orchestration Score', data.scorer || [], { scales: { r: { beginAtZero: true, suggestedMax: 1, ticks: { stepSize: 0.25 } } } }, 'radar');

                // tsr: binary-like 0/1 values (or 0..1), keep same scale 0..1
                createChart('chart-iter-' + key + '-tsr', 'User Success Rate', data.tsr || [], { scales: { r: { beginAtZero: true, suggestedMax: 1, ticks: { stepSize: 0.25 } }, y: { beginAtZero: true, suggestedMax: 1, ticks: { callback: function(v){ return (v*100).toFixed(1) + ' %'; } } } } }, 'radar');

                // latency: numeric seconds - set suggestedMax based on max value
                var latencyArr = data.latency || [];
                var maxLatency = 1;
                if(latencyArr.length) {
                    maxLatency = Math.max.apply(null, latencyArr.map(function(v){ return Number(v) || 0; }));
                }
                var suggested = Math.ceil(maxLatency * 1.2) || 1;
                // pass both r (for radar) and y (for bar) scale hints
                createChart('chart-iter-' + key + '-latency', 'Orchestration Latency (s)', latencyArr, { scales: { r: { beginAtZero: true, suggestedMax: suggested, ticks: { stepSize: Math.max(1, Math.ceil(suggested/4)) } }, y: { beginAtZero: true, suggestedMax: suggested } } }, null);

            }catch(e){
                console.warn('chart render error', e);
            }
        });
    }

    // enhanced markdown renderer for small docs and final outputs
    function renderMarkdown(md){
        if(md === null || md === undefined) return '';
        let s = String(md);
        // escape HTML first
        s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // extract fenced code blocks and replace with placeholders
        const codeBlocks = [];
        s = s.replace(/```([\s\S]*?)```/g, function(m, code){
            codeBlocks.push(code);
            return `@@CODEBLOCK_${codeBlocks.length-1}@@`;
        });

        // split into paragraphs/blocks by blank line
        const parts = s.split(/\n\s*\n/);
        const out = [];

        function inlineFormat(str){
            // inline code `code`
            str = str.replace(/`([^`]+)`/g, function(m, c){ return '<code class="bg-gray-100 px-1 rounded">' + c + '</code>'; });
            // bold **text**
            str = str.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // italic *text*
            str = str.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            // simple link [text](url)
            str = str.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-indigo-600 hover:underline" target="_blank">$1</a>');
            return str;
        }

        parts.forEach(function(part){
            const t = part.trim();
            if(!t) return;
            // heading
            const h = t.match(/^#{1,6}\s+(.*)/);
            if(h){
                const level = Math.min(6, (t.match(/^#+/)[0] || '').length);
                out.push(`<h${level} class="text-gray-800 font-semibold mt-2 mb-2">${inlineFormat(h[1])}</h${level}>`);
                return;
            }

            // unordered list (lines starting with - or *)
            const lines = t.split('\n');
            const isUL = lines.every(l => /^\s*[-*]\s+/.test(l));
            const isOL = lines.every(l => /^\s*\d+\.\s+/.test(l));
            if(isUL){
                out.push('<ul class="list-disc ml-5 mb-2">' + lines.map(l => '<li>' + inlineFormat(l.replace(/^\s*[-*]\s+/, '')) + '</li>').join('') + '</ul>');
                return;
            }
            if(isOL){
                out.push('<ol class="list-decimal ml-5 mb-2">' + lines.map(l => '<li>' + inlineFormat(l.replace(/^\s*\d+\.\s+/, '')) + '</li>').join('') + '</ol>');
                return;
            }

            // otherwise paragraph - preserve single newlines as <br>
            out.push('<p class="text-gray-700 mb-2">' + inlineFormat(t.replace(/\n/g, '<br>')) + '</p>');
        });

        let html = out.join('\n');

        // restore code blocks
        html = html.replace(/@@CODEBLOCK_(\d+)@@/g, function(m, idx){
            const code = codeBlocks[Number(idx)] || '';
            const esc = String(code).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            return `<pre class="bg-gray-100 p-3 rounded mb-3"><code>${esc}</code></pre>`;
        });

        return html;
    }

    function renderFinalOutputs(){
        document.querySelectorAll('.final-output').forEach(container => {
            try{
                const pre = container.querySelector('.final-output-raw');
                const renderDiv = container.querySelector('.final-output-render');
                if(!pre || !renderDiv) return;
                const raw = pre.textContent || '';
                const html = renderMarkdown(raw);
                renderDiv.innerHTML = html;
                // hide raw pre (already display:none)
            }catch(e){
                console.warn('render final output error', e);
            }
        });
    }

    function renderEvaluatorDocs(){
        try{
            const pre = document.querySelector('.evaluator-docs-raw');
            const target = document.querySelector('.evaluator-docs-render');
            if(!pre || !target) return;
            const raw = pre.textContent || '';
            const html = renderMarkdown(raw);
            // wrap the rendered markdown in a styled container for better visual layout
            target.innerHTML = `<div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">` +
                `<div class="col-span-1 md:col-span-2 lg:col-span-3 mb-2"><div class="bg-gradient-to-r from-gray-50 to-white p-4 rounded border border-gray-100 shadow-sm">` +
                `<h3 class="text-lg font-semibold text-gray-800">Evaluator Metrics</h3>` +
                `<p class="text-sm text-gray-600 mt-1">Quick summary of metrics used by the benchmark evaluator and how scores are computed.</p>` +
                `</div></div>` +
                `<div class="prose max-w-none col-span-1 md:col-span-2">${html}</div>` +
                `</div>`;
        }catch(e){ console.warn('render evaluator docs error', e); }
    }

    function setupDetailsToggles(){
        // Collapse all details rows
        function collapseAll(){
            document.querySelectorAll('.details-row').forEach(r => r.classList.add('hidden'));
        }

        document.querySelectorAll('.details-toggle').forEach(btn => {
            btn.addEventListener('click', function(e){
                const targetId = btn.getAttribute('data-target');
                if(!targetId) return;
                const target = document.getElementById(targetId);
                if(!target) return;

                const isHidden = target.classList.contains('hidden');
                // collapse others first
                collapseAll();
                if(isHidden){
                    target.classList.remove('hidden');
                    // render markdown for newly visible container inside target
                    try{
                        const fo = target.querySelector('.final-output');
                        if(fo){
                            const pre = fo.querySelector('.final-output-raw');
                            const renderDiv = fo.querySelector('.final-output-render');
                            if(pre && renderDiv){
                                renderDiv.innerHTML = renderMarkdown(pre.textContent || '');
                            }
                        }
                    }catch(err){ console.warn('render details error', err); }
                }else{
                    target.classList.add('hidden');
                }
            });
        });

        // Ensure details collapse when iteration filter changes
        const select = document.getElementById('iteration-select');
        if(select){
            select.addEventListener('change', function(){
                collapseAll();
            });
        }
    }

    document.addEventListener('DOMContentLoaded', function(){
        const select = document.getElementById('iteration-select');
        if(select){
            select.addEventListener('change', (e)=>{
                applyFilter(e.target.value);
            });
            // initialize
            applyFilter(select.value || 'all');
        }
        // render charts after DOM ready
        renderCharts();
        // render markdown-based final outputs (for any always-visible final outputs)
        renderFinalOutputs();
        // render evaluator docs markdown (if present)
        renderEvaluatorDocs();
        // setup per-row details toggles (collapsible final output rows)
        setupDetailsToggles();
    });
})();
