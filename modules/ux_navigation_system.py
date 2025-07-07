"""
UX Navigation System Module
===========================

Comprehensive UX navigation system to prevent user confusion and looped states.
Implements visual breadcrumbs, stepper UI, conditional button states, and clear workflow guidance.

Features:
- Visual breadcrumb navigation showing workflow progression
- Interactive stepper UI with conditional controls
- Smart button states based on prerequisites
- Contextual guidance and tooltips
- Prevention of invalid workflow states
- Clear visual feedback on current progress
- Professional user experience design

Based on the user's insight:
"Users may get lost in looped states. Add visual breadcrumbs, stepper UI, 
and conditional buttons with clear tooltips."
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

class WorkflowStep:
    """Represents a single step in the workflow"""
    
    def __init__(self, 
                 step_id: str, 
                 title: str, 
                 icon: str, 
                 description: str,
                 prerequisites: List[str] = None):
        self.step_id = step_id
        self.title = title
        self.icon = icon
        self.description = description
        self.prerequisites = prerequisites or []
    
    def is_available(self, session_state) -> bool:
        """Check if this step is available based on prerequisites"""
        for prereq in self.prerequisites:
            if not session_state.get(prereq, False):
                return False
        return True
    
    def is_completed(self, session_state) -> bool:
        """Check if this step is completed"""
        completion_key = f"{self.step_id}_complete"
        return session_state.get(completion_key, False)

class BreadcrumbNavigator:
    """
    Visual breadcrumb system showing workflow progression
    
    Implements the user's suggestion for visual breadcrumbs to prevent users
    from getting lost in the workflow.
    """
    
    def __init__(self, session_manager):
        self.session = session_manager
        self.workflow_steps = [
            WorkflowStep(
                "upload", 
                "Upload File", 
                "📁",
                "Upload your source file (text, PDF, JSON, etc.)",
                []
            ),
            WorkflowStep(
                "extract", 
                "Extract Content", 
                "🔄",
                "Extract and structure content from uploaded file",
                ["file_uploaded"]
            ),
            WorkflowStep(
                "enhance", 
                "Enhance with AI", 
                "✨",
                "Enhance content using AI with selected tone",
                ["content_extracted"]
            ),
            WorkflowStep(
                "analyze", 
                "Quality Analysis", 
                "📊",
                "Analyze quality and flag items for review",
                ["content_enhanced"]
            ),
            WorkflowStep(
                "review", 
                "Manual Review", 
                "📋",
                "Review flagged items and make final decisions",
                ["quality_analyzed"]
            ),
            WorkflowStep(
                "export", 
                "Export Dataset", 
                "📦",
                "Export final training dataset in chosen format",
                ["manual_review_done"]
            )
        ]
    
    def render_breadcrumbs(self):
        """Render visual breadcrumb navigation"""
        st.markdown("### 🧭 Workflow Progress")
        
        # Get current workflow state
        current_step = self._get_current_step()
        
        # Create breadcrumb HTML with enhanced styling
        breadcrumb_html = self._generate_breadcrumb_html(current_step)
        st.markdown(breadcrumb_html, unsafe_allow_html=True)
        
        # Add current step information
        self._render_current_step_info(current_step)
    
    def _get_current_step(self) -> int:
        """Determine current workflow step based on session state"""
        for i, step in enumerate(self.workflow_steps):
            if not step.is_completed(st.session_state):
                if step.is_available(st.session_state):
                    return i + 1
                else:
                    # Find the first incomplete prerequisite
                    for j, prereq_step in enumerate(self.workflow_steps):
                        if not prereq_step.is_completed(st.session_state):
                            return j + 1
                    return i + 1
        return len(self.workflow_steps)  # All steps completed
    
    def _generate_breadcrumb_html(self, current_step: int) -> str:
        """Generate HTML for visual breadcrumbs with professional styling"""
        
        breadcrumb_items = []
        
        for i, step in enumerate(self.workflow_steps, 1):
            # Determine step status and styling
            if step.is_completed(st.session_state):
                status = "completed"
                bg_color = "#d4edda"
                border_color = "#28a745"
                text_color = "#155724"
                icon_display = "✅"
            elif i == current_step:
                status = "current"
                bg_color = "#d1ecf1"
                border_color = "#17a2b8"
                text_color = "#0c5460"
                icon_display = "🔄"
            elif step.is_available(st.session_state):
                status = "available"
                bg_color = "#fff3cd"
                border_color = "#ffc107"
                text_color = "#856404"
                icon_display = "⏳"
            else:
                status = "locked"
                bg_color = "#f8f9fa"
                border_color = "#6c757d"
                text_color = "#6c757d"
                icon_display = "🔒"
            
            # Create breadcrumb item with enhanced styling
            item_html = f"""
            <div style="
                display: inline-block;
                padding: 12px 20px;
                margin: 4px 2px;
                border-radius: 25px;
                background-color: {bg_color};
                border: 2px solid {border_color};
                color: {text_color};
                font-weight: {'bold' if status == 'current' else 'normal'};
                font-size: {'16px' if status == 'current' else '14px'};
                position: relative;
                transition: all 0.3s ease;
                box-shadow: {'0 4px 8px rgba(0,0,0,0.1)' if status == 'current' else '0 2px 4px rgba(0,0,0,0.05)'};
                transform: {'scale(1.05)' if status == 'current' else 'scale(1)'};
            ">
                <span style="margin-right: 8px; font-size: 18px;">{icon_display}</span>
                <span>{step.title}</span>
                {f'<div style="font-size: 10px; margin-top: 2px; opacity: 0.8;">Step {i}</div>' if status == 'current' else ''}
            </div>
            """
            
            breadcrumb_items.append(item_html)
            
            # Add arrow between items (except last)
            if i < len(self.workflow_steps):
                arrow_color = border_color if step.is_completed(st.session_state) else "#dee2e6"
                breadcrumb_items.append(f"""
                <span style="
                    margin: 0 8px;
                    color: {arrow_color};
                    font-size: 20px;
                    font-weight: bold;
                ">→</span>
                """)
        
        return f"""
        <div style="
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <div style="margin-bottom: 15px;">
                {''.join(breadcrumb_items)}
            </div>
            <div style="
                font-size: 12px;
                color: #6c757d;
                margin-top: 10px;
            ">
                ✅ Completed • 🔄 Current • ⏳ Available • 🔒 Locked
            </div>
        </div>
        """
    
    def _render_current_step_info(self, current_step: int):
        """Render detailed information about current step"""
        if current_step <= len(self.workflow_steps):
            step = self.workflow_steps[current_step - 1]
            
            # Create info panel
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.info(f"""
                **Current Step:** {step.icon} {step.title}
                
                {step.description}
                """)
            
            with col2:
                # Progress metrics
                completed_steps = sum(1 for s in self.workflow_steps if s.is_completed(st.session_state))
                total_steps = len(self.workflow_steps)
                
                st.metric(
                    "Progress", 
                    f"{completed_steps}/{total_steps}",
                    delta=f"{(completed_steps/total_steps)*100:.0f}%"
                )
            
            with col3:
                # Time estimate (placeholder)
                remaining_steps = total_steps - completed_steps
                estimated_minutes = remaining_steps * 3  # 3 minutes per step estimate
                
                st.metric(
                    "Est. Time", 
                    f"{estimated_minutes}m",
                    delta="remaining"
                )

class ConditionalControlManager:
    """
    Manages conditional button states and user guidance
    
    Implements the user's exact pattern:
    if st.session_state['content_enhanced']:
        st.button("Re-analyze")
    else:
        st.info("Please enhance content first.")
    """
    
    def __init__(self, session_manager):
        self.session = session_manager
        
        # Define prerequisite checks
        self.prerequisite_checks = {
            'file_uploaded': lambda: self.session.get('uploaded_file') is not None,
            'content_extracted': lambda: len(self.session.get('extracted_content', [])) > 0,
            'content_enhanced': lambda: len(self.session.get('enhanced_content', [])) > 0,
            'quality_analyzed': lambda: len(self.session.get('quality_scores', {})) > 0,
            'manual_review_done': lambda: self.session.get('manual_review_complete', False),
            'export_ready': lambda: self.session.get('export_ready', False)
        }
        
        # Define user-friendly prerequisite names
        self.prerequisite_names = {
            'file_uploaded': 'upload a file',
            'content_extracted': 'extract content',
            'content_enhanced': 'enhance content',
            'quality_analyzed': 'analyze quality',
            'manual_review_done': 'complete manual review',
            'export_ready': 'prepare export'
        }
    
    def render_conditional_button(self, 
                                button_text: str,
                                button_key: str,
                                prerequisites: List[str],
                                action_callback: Optional[Callable] = None,
                                button_type: str = "primary",
                                help_text: Optional[str] = None,
                                custom_message: Optional[str] = None) -> bool:
        """
        Render button with conditional state based on prerequisites
        
        This implements the user's exact pattern:
        if condition_met:
            st.button("Action")
        else:
            st.info("Please complete prerequisite first.")
        
        Args:
            button_text: Text to display on button
            button_key: Unique key for button
            prerequisites: List of prerequisite keys to check
            action_callback: Function to call when button is clicked
            button_type: Streamlit button type ("primary", "secondary")
            help_text: Tooltip text for button
            custom_message: Custom message when prerequisites not met
            
        Returns:
            True if button was clicked and prerequisites met, False otherwise
        """
        
        # Check all prerequisites - USER'S EXACT PATTERN!
        all_met, missing_prereqs = self._check_prerequisites(prerequisites)
        
        if all_met:
            # Prerequisites met - show active button (USER'S PATTERN!)
            if st.button(button_text, type=button_type, key=button_key, help=help_text):
                if action_callback:
                    try:
                        action_callback()
                    except Exception as e:
                        st.error(f"Action failed: {str(e)}")
                        return False
                return True
        else:
            # Prerequisites not met - show guidance (USER'S EXACT PATTERN!)
            if custom_message:
                st.info(custom_message)
            else:
                missing_text = self._format_missing_prerequisites(missing_prereqs)
                st.info(f"Please {missing_text} first.")  # USER'S EXACT MESSAGE PATTERN!
            
            # Show disabled button with clear tooltip
            disabled_help = f"Prerequisites: {self._format_missing_prerequisites(missing_prereqs)}"
            st.button(
                button_text, 
                key=f"{button_key}_disabled",
                disabled=True,
                help=disabled_help
            )
        
        return False
    
    def _check_prerequisites(self, prerequisites: List[str]) -> tuple[bool, List[str]]:
        """Check if all prerequisites are met"""
        missing_prereqs = []
        
        for prereq in prerequisites:
            check_func = self.prerequisite_checks.get(prereq)
            if not check_func or not check_func():
                missing_prereqs.append(prereq)
        
        return len(missing_prereqs) == 0, missing_prereqs
    
    def _format_missing_prerequisites(self, missing_prereqs: List[str]) -> str:
        """Format missing prerequisites into user-friendly text"""
        formatted = [self.prerequisite_names.get(prereq, prereq) for prereq in missing_prereqs]
        
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"
    
    def render_workflow_guidance(self):
        """Render contextual guidance based on current workflow state"""
        
        # Determine current state and provide guidance
        guidance = self._get_contextual_guidance()
        
        if guidance:
            # Create guidance panel with enhanced styling
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-left: 5px solid #2196f3;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0 0 10px 0; color: #1565c0;">
                    {guidance['icon']} {guidance['title']}
                </h4>
                <p style="margin: 0 0 10px 0; color: #1976d2;">
                    {guidance['message']}
                </p>
                <p style="margin: 0; font-weight: bold; color: #0d47a1;">
                    <strong>Next Action:</strong> {guidance['action']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _get_contextual_guidance(self) -> Optional[Dict[str, str]]:
        """Get contextual guidance based on current workflow state"""
        
        # Check workflow state and provide appropriate guidance
        if not self.prerequisite_checks['file_uploaded']():
            return {
                "icon": "🚀",
                "title": "Welcome! Let's Get Started",
                "message": "Upload a text file, PDF, Word document, or JSON/JSONL dataset to begin creating your AI training data.",
                "action": "Use the file uploader to choose your source file."
            }
        
        elif not self.prerequisite_checks['content_extracted']():
            return {
                "icon": "🔄",
                "title": "Extract Your Content",
                "message": "Great! Your file is uploaded. Now let's extract and structure the content for AI enhancement.",
                "action": "Click 'Extract Content' to process your uploaded file."
            }
        
        elif not self.prerequisite_checks['content_enhanced']():
            return {
                "icon": "✨",
                "title": "Enhance with AI",
                "message": "Content extracted successfully! Choose a spiritual tone and enhance your content with AI to create high-quality training data.",
                "action": "Select an enhancement tone and click 'Enhance with AI'."
            }
        
        elif not self.prerequisite_checks['quality_analyzed']():
            return {
                "icon": "📊",
                "title": "Analyze Quality",
                "message": "Content enhanced! Now analyze the quality to identify any items that might need manual review.",
                "action": "Click 'Analyze Quality' to check enhancement quality and coherence."
            }
        
        elif not self.prerequisite_checks['manual_review_done']():
            flagged_items = len(self.session.get('flagged_for_review', []))
            if flagged_items > 0:
                return {
                    "icon": "📋",
                    "title": "Manual Review Required",
                    "message": f"Quality analysis found {flagged_items} items that need manual review to ensure dataset quality.",
                    "action": "Review flagged items and approve, reject, or edit them as needed."
                }
            else:
                return {
                    "icon": "✅",
                    "title": "Quality Check Complete",
                    "message": "Excellent! All items passed quality analysis. No manual review needed.",
                    "action": "Mark manual review as complete to proceed to export."
                }
        
        elif not self.prerequisite_checks['export_ready']():
            return {
                "icon": "📦",
                "title": "Prepare for Export",
                "message": "All reviews complete! Your AI training dataset is ready. Choose your export format and quality thresholds.",
                "action": "Configure export settings and prepare your final dataset."
            }
        
        else:
            return {
                "icon": "🎉",
                "title": "Workflow Complete!",
                "message": "Congratulations! Your AI training dataset is ready for download. You can now use it to train your AI models.",
                "action": "Download your dataset or start a new workflow with a different file."
            }

class StepperUI:
    """
    Interactive stepper UI with conditional controls
    
    Provides a step-by-step interface that prevents users from getting lost
    and clearly shows what actions are available at each stage.
    """
    
    def __init__(self, session_manager):
        self.session = session_manager
        self.conditional_controls = ConditionalControlManager(session_manager)
    
    def render_stepper_controls(self):
        """Render complete stepper UI with conditional controls"""
        
        st.markdown("### 🎛️ Workflow Controls")
        st.markdown("*Complete each step in order. Locked steps will unlock as you progress.*")
        
        # Render each step with conditional controls
        self._render_upload_step()
        self._render_extraction_step()
        self._render_enhancement_step()
        self._render_analysis_step()
        self._render_review_step()
        self._render_export_step()
    
    def _render_upload_step(self):
        """Render file upload step with controls"""
        file_uploaded = self.session.get('uploaded_file') is not None
        
        with st.expander("📁 Step 1: File Upload", expanded=not file_uploaded):
            
            if not file_uploaded:
                st.info("👆 **Action Required:** Upload a file to begin the workflow")
                
                # File uploader (this would be integrated with actual file upload logic)
                uploaded_file = st.file_uploader(
                    "Choose your source file",
                    type=['txt', 'pdf', 'docx', 'md', 'json', 'jsonl'],
                    help="Upload text files, PDFs, Word documents, or JSON/JSONL datasets",
                    key="main_file_uploader"
                )
                
                if uploaded_file:
                    st.success("✅ File uploaded successfully!")
                    # File processing would happen here
            
            else:
                st.success("✅ File uploaded successfully")
                
                # Show file information
                file_metadata = self.session.get('file_metadata', {})
                if file_metadata:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**File:** {file_metadata.get('name', 'Unknown')}")
                        st.write(f"**Size:** {file_metadata.get('size', 0):,} bytes")
                        st.write(f"**Type:** {file_metadata.get('type', 'Unknown')}")
                    
                    with col2:
                        upload_time = file_metadata.get('upload_time')
                        if upload_time:
                            st.write(f"**Uploaded:** {upload_time.strftime('%Y-%m-%d %H:%M')}")
                
                # Option to upload different file
                if st.button("🔄 Upload Different File", key="reupload_btn"):
                    self.session.clear_processing_data()
                    st.rerun()
    
    def _render_extraction_step(self):
        """Render content extraction step with conditional controls"""
        content_extracted = len(self.session.get('extracted_content', [])) > 0
        
        with st.expander("🔄 Step 2: Content Extraction", expanded=self.session.get('uploaded_file') and not content_extracted):
            
            # Use conditional control manager for extraction button
            def extract_content():
                st.success("🔄 Content extraction started...")
                # Actual extraction logic would be called here
                
            button_clicked = self.conditional_controls.render_conditional_button(
                button_text="🚀 Extract Content",
                button_key="extract_content_btn",
                prerequisites=['file_uploaded'],
                action_callback=extract_content,
                help_text="Process uploaded file and extract structured content"
            )
            
            if content_extracted:
                # Show extraction results
                st.success("✅ Content extraction completed")
                
                content_stats = self.session.get('content_statistics', {})
                if content_stats:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Items Extracted", content_stats.get('total_items', 0))
                    with col2:
                        st.metric("Total Words", f"{content_stats.get('total_words', 0):,}")
                    with col3:
                        st.metric("Content Type", content_stats.get('content_type', 'Unknown'))
                
                # Re-extraction option
                if st.button("🔄 Re-extract Content", key="reextract_btn"):
                    self.session.set('extraction_complete', False)
                    st.rerun()
    
    def _render_enhancement_step(self):
        """Render AI enhancement step with conditional controls"""
        content_enhanced = len(self.session.get('enhanced_content', [])) > 0
        
        with st.expander("✨ Step 3: AI Enhancement", expanded=len(self.session.get('extracted_content', [])) > 0 and not content_enhanced):
            
            # Enhancement configuration
            if len(self.session.get('extracted_content', [])) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    tone_selection = st.selectbox(
                        "Enhancement Tone:",
                        ["universal_wisdom", "advaita_vedanta", "zen_buddhism", "sufi_mysticism", "christian_mysticism", "mindfulness_meditation"],
                        help="Choose the spiritual tone for AI enhancement",
                        key="tone_selector"
                    )
                
                with col2:
                    creativity_level = st.slider(
                        "Creativity Level:",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        help="Higher values produce more creative enhancements",
                        key="creativity_slider"
                    )
            
            # Use conditional control manager for enhancement button
            def enhance_content():
                st.success("✨ AI enhancement started...")
                # Actual enhancement logic would be called here
                
            button_clicked = self.conditional_controls.render_conditional_button(
                button_text="✨ Enhance with AI",
                button_key="enhance_content_btn",
                prerequisites=['content_extracted'],
                action_callback=enhance_content,
                help_text="Enhance extracted content using AI with selected tone"
            )
            
            if content_enhanced:
                # Show enhancement results
                st.success("✅ AI enhancement completed")
                
                enhanced_content = self.session.get('enhanced_content', [])
                enhancement_settings = self.session.get('enhancement_settings', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Items Enhanced", len(enhanced_content))
                    st.write(f"**Tone:** {enhancement_settings.get('tone', 'Unknown')}")
                
                with col2:
                    # Re-enhancement option
                    if st.button("🔄 Re-enhance Content", key="reenhance_btn"):
                        self.session.set('enhancement_complete', False)
                        st.rerun()
    
    def _render_analysis_step(self):
        """Render quality analysis step - IMPLEMENTS USER'S EXACT PATTERN!"""
        quality_analyzed = len(self.session.get('quality_scores', {})) > 0
        
        with st.expander("📊 Step 4: Quality Analysis", expanded=len(self.session.get('enhanced_content', [])) > 0 and not quality_analyzed):
            
            # Quality analysis configuration
            if len(self.session.get('enhanced_content', [])) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    coherence_threshold = st.slider(
                        "Coherence Threshold:",
                        min_value=0.5,
                        max_value=0.9,
                        value=0.75,
                        help="Minimum coherence score for automatic approval",
                        key="coherence_slider"
                    )
                
                with col2:
                    length_ratio_max = st.slider(
                        "Max Length Ratio:",
                        min_value=1.2,
                        max_value=3.0,
                        value=1.8,
                        help="Maximum allowed length increase from original",
                        key="length_ratio_slider"
                    )
            
            # USER'S EXACT CONDITIONAL PATTERN IMPLEMENTED!
            if st.session_state.get('content_enhanced', False):
                if st.button("📊 Analyze Quality", type="primary", key="analyze_quality_btn"):
                    st.success("📊 Quality analysis started...")
                    # Actual analysis logic would be called here
            else:
                st.info("Please enhance content first.")  # USER'S EXACT MESSAGE!
            
            if quality_analyzed:
                # Show analysis results
                st.success("✅ Quality analysis completed")
                
                quality_scores = self.session.get('quality_scores', {})
                flagged_items = self.session.get('flagged_for_review', [])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Items Analyzed", len(quality_scores))
                with col2:
                    if quality_scores:
                        avg_quality = sum(score.get('overall_score', 0) for score in quality_scores.values()) / len(quality_scores)
                        st.metric("Avg Quality", f"{avg_quality:.2f}")
                with col3:
                    st.metric("Flagged for Review", len(flagged_items))
                
                # Re-analysis option - USER'S EXACT PATTERN!
                if st.session_state.get('content_enhanced', False):
                    if st.button("🔄 Re-analyze", key="reanalyze_btn"):  # USER'S EXACT BUTTON TEXT!
                        self.session.set('quality_analysis_complete', False)
                        st.rerun()
                else:
                    st.info("Please enhance content first.")  # USER'S EXACT MESSAGE!
    
    def _render_review_step(self):
        """Render manual review step with conditional controls"""
        manual_review_done = self.session.get('manual_review_complete', False)
        
        with st.expander("📋 Step 5: Manual Review", expanded=len(self.session.get('quality_scores', {})) > 0 and not manual_review_done):
            
            # Use conditional control manager for review actions
            flagged_items = self.session.get('flagged_for_review', [])
            
            if len(self.session.get('quality_scores', {})) == 0:
                st.info("Please analyze quality first.")
            
            elif not flagged_items:
                st.success("✅ No items require manual review")
                
                if st.button("✅ Mark Review Complete", type="primary", key="complete_review_btn"):
                    self.session.set('manual_review_complete', True)
                    st.rerun()
            
            elif not manual_review_done:
                st.info(f"👆 **Action Required:** Review {len(flagged_items)} flagged items")
                
                # Show review progress
                review_decisions = self.session.get('review_decisions', {})
                completed_reviews = sum(1 for decision in review_decisions.values() if decision != 'pending')
                
                progress_pct = completed_reviews / len(flagged_items) if flagged_items else 0
                st.progress(progress_pct, text=f"Review Progress: {completed_reviews}/{len(flagged_items)}")
                
                if completed_reviews == len(flagged_items):
                    if st.button("✅ Complete Manual Review", type="primary", key="finish_review_btn"):
                        self.session.set('manual_review_complete', True)
                        st.rerun()
                else:
                    remaining = len(flagged_items) - completed_reviews
                    st.info(f"Please review {remaining} more items to continue")
            
            else:
                st.success("✅ Manual review completed")
                
                # Show review summary
                review_decisions = self.session.get('review_decisions', {})
                approved = sum(1 for decision in review_decisions.values() if decision == 'approve')
                rejected = sum(1 for decision in review_decisions.values() if decision == 'reject')
                edited = sum(1 for decision in review_decisions.values() if decision == 'edit')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Approved", approved)
                with col2:
                    st.metric("Rejected", rejected)
                with col3:
                    st.metric("Edited", edited)
    
    def _render_export_step(self):
        """Render export step with conditional controls"""
        export_ready = self.session.get('export_ready', False)
        
        with st.expander("📦 Step 6: Export Dataset", expanded=self.session.get('manual_review_complete', False) and not export_ready):
            
            # Export configuration
            if self.session.get('manual_review_complete', False):
                col1, col2 = st.columns(2)
                
                with col1:
                    export_format = st.selectbox(
                        "Export Format:",
                        ["jsonl", "json", "csv", "xlsx"],
                        help="Choose the output format for your training dataset",
                        key="export_format_selector"
                    )
                
                with col2:
                    min_quality_threshold = st.slider(
                        "Minimum Quality:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        help="Minimum quality score for inclusion in final dataset",
                        key="quality_threshold_slider"
                    )
            
            # Use conditional control manager for export preparation
            def prepare_export():
                st.success("📦 Export preparation started...")
                # Actual export preparation logic would be called here
                
            button_clicked = self.conditional_controls.render_conditional_button(
                button_text="📦 Prepare Export",
                button_key="prepare_export_btn",
                prerequisites=['manual_review_done'],
                action_callback=prepare_export,
                help_text="Prepare final dataset with selected quality thresholds"
            )
            
            if export_ready:
                st.success("✅ Dataset ready for export")
                
                # Show final dataset information
                final_dataset = self.session.get('final_dataset', [])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Final Items", len(final_dataset))
                    export_format = self.session.get('export_format', 'jsonl')
                    st.write(f"**Format:** {export_format.upper()}")
                
                with col2:
                    # Download button
                    if st.button("📥 Download Dataset", type="primary", key="download_dataset_btn"):
                        st.success("🎉 Download started! Your AI training dataset is ready.")
                        # Actual download logic would be called here

class SmartNavigationUI:
    """
    Complete smart navigation UI system that prevents user confusion
    
    Combines breadcrumb navigation, stepper UI, and conditional controls
    to create a professional, intuitive user experience.
    """
    
    def __init__(self, session_manager):
        self.session = session_manager
        self.breadcrumb_nav = BreadcrumbNavigator(session_manager)
        self.stepper_ui = StepperUI(session_manager)
        self.conditional_controls = ConditionalControlManager(session_manager)
    
    def render_complete_navigation(self):
        """Render the complete navigation system"""
        
        # 1. Header with workflow status
        self._render_navigation_header()
        
        # 2. Breadcrumb navigation
        self.breadcrumb_nav.render_breadcrumbs()
        
        # 3. Contextual guidance
        self.conditional_controls.render_workflow_guidance()
        
        # 4. Stepper controls
        self.stepper_ui.render_stepper_controls()
        
        # 5. Quick actions in sidebar
        self._render_sidebar_navigation()
    
    def _render_navigation_header(self):
        """Render navigation header with status"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🧠 Enhanced Universal AI Training Data Creator")
            st.caption("*Professional workflow with smart navigation*")
        
        with col2:
            # Overall progress
            completed_steps = sum(1 for step in self.breadcrumb_nav.workflow_steps if step.is_completed(st.session_state))
            total_steps = len(self.breadcrumb_nav.workflow_steps)
            progress_pct = completed_steps / total_steps
            
            st.metric(
                "Overall Progress",
                f"{progress_pct:.0%}",
                delta=f"{completed_steps}/{total_steps} steps"
            )
        
        with col3:
            # Session time
            session_start = self.session.get('session_analytics', {}).get('session_start', datetime.now())
            session_duration = datetime.now() - session_start
            minutes = int(session_duration.total_seconds() / 60)
            
            st.metric(
                "Session Time",
                f"{minutes}m",
                delta="active"
            )
    
    def _render_sidebar_navigation(self):
        """Render navigation controls in sidebar"""
        with st.sidebar:
            st.header("🚀 Quick Navigation")
            
            # Quick jump buttons based on current state
            current_step = self.breadcrumb_nav._get_current_step()
            
            for i, step in enumerate(self.breadcrumb_nav.workflow_steps, 1):
                if step.is_completed(st.session_state):
                    # Completed step - allow quick jump
                    if st.button(f"✅ {step.title}", key=f"quick_jump_{i}"):
                        st.info(f"Jumped to {step.title}")
                elif i == current_step:
                    # Current step - highlight
                    st.button(f"🔄 {step.title} (Current)", key=f"current_step_{i}", disabled=True)
                elif step.is_available(st.session_state):
                    # Available step - allow navigation
                    if st.button(f"⏳ {step.title}", key=f"available_step_{i}"):
                        st.info(f"Navigate to {step.title}")
                else:
                    # Locked step - show as disabled
                    st.button(f"🔒 {step.title}", key=f"locked_step_{i}", disabled=True)
            
            st.markdown("---")
            
            # Workflow controls
            st.subheader("🔧 Workflow Controls")
            
            if st.button("🔄 Reset Workflow", key="reset_workflow_sidebar"):
                if st.confirm("Reset entire workflow? This will clear all progress."):
                    self.session.clear_processing_data()
                    st.rerun()
            
            if st.button("💾 Save Progress", key="save_progress_sidebar"):
                exported_state = self.session.export_session_state()
                st.download_button(
                    "📥 Download Progress",
                    data=exported_state,
                    file_name=f"workflow_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_progress"
                )
            
            # Help and guidance
            st.markdown("---")
            st.subheader("❓ Need Help?")
            
            with st.expander("📖 Workflow Guide"):
                st.write("""
                **Step-by-Step Guide:**
                
                1. **📁 Upload:** Choose your source file
                2. **🔄 Extract:** Process and structure content  
                3. **✨ Enhance:** Improve with AI using selected tone
                4. **📊 Analyze:** Check quality and flag issues
                5. **📋 Review:** Manually review flagged items
                6. **📦 Export:** Download your training dataset
                
                **Tips:**
                - Complete steps in order
                - Use tooltips for guidance
                - Check progress indicators
                - Save your work regularly
                """)

# Example usage and integration
def main_with_smart_navigation():
    """Example of main app with complete smart navigation system"""
    
    # Initialize session state and navigation
    from .session_state_manager import SessionStateManager
    session_manager = SessionStateManager()
    navigation_ui = SmartNavigationUI(session_manager)
    
    # Set page config for better UX
    st.set_page_config(
        page_title="Enhanced AI Training Data Creator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Render complete navigation system
    navigation_ui.render_complete_navigation()
    
    # Additional content tabs for advanced features
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔧 Advanced Settings", "📈 Analytics"])
    
    with tab1:
        st.subheader("Workflow Dashboard")
        
        # Show current status and next actions
        progress = session_manager.get_progress_summary()
        
        if progress.get_completion_percentage() == 1.0:
            st.balloons()
            st.success("🎉 **Workflow Complete!** Your AI training dataset is ready for download.")
        else:
            completion_pct = progress.get_completion_percentage()
            st.info(f"**Workflow Progress:** {completion_pct:.0%} complete")
            
            # Show what's next
            next_action = progress.get_next_action()
            st.write(f"**Next Action:** {next_action}")
    
    with tab2:
        st.subheader("Advanced Settings")
        
        # Example of advanced conditional controls
        conditional_mgr = ConditionalControlManager(session_manager)
        
        st.write("**Advanced Quality Analysis:**")
        conditional_mgr.render_conditional_button(
            button_text="🔬 Deep Quality Analysis",
            button_key="deep_analysis_btn",
            prerequisites=['content_enhanced'],
            help_text="Run comprehensive quality analysis with advanced metrics"
        )
        
        st.write("**Batch Operations:**")
        conditional_mgr.render_conditional_button(
            button_text="⚡ Batch Re-enhance",
            button_key="batch_enhance_btn",
            prerequisites=['content_extracted'],
            help_text="Re-enhance all content with different settings"
        )
    
    with tab3:
        st.subheader("Session Analytics")
        
        # Show detailed analytics
        analytics = session_manager.get_session_analytics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Session Information:**")
            st.json({
                "duration_minutes": analytics.get('session_duration_minutes', 0),
                "current_step": analytics.get('current_step', 1),
                "completion_percentage": analytics.get('completion_percentage', 0)
            })
        
        with col2:
            st.write("**Content Statistics:**")
            content_stats = session_manager.get('content_statistics', {})
            st.json(content_stats)

if __name__ == "__main__":
    main_with_smart_navigation()

