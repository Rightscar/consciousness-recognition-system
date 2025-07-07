"""
Quality Scoring Dashboard
========================

Live quality score dashboard for reviewer feedback and comprehensive
quality assessment with real-time metrics and visual indicators.

Features:
- Live quality scoring with real-time updates
- Visual metrics dashboard with color coding
- Comprehensive quality assessment
- Reviewer feedback integration
- Quality data export for fine-tuning
- Historical quality tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from modules.logger import get_logger, log_event

class QualityScoringDashboard:
    """
    Comprehensive quality scoring dashboard
    
    Features:
    - Real-time quality metrics
    - Visual scoring dashboard
    - Reviewer feedback system
    - Quality data collection
    - Historical tracking
    - Export functionality
    """
    
    def __init__(self):
        self.logger = get_logger("quality_dashboard")
        
        # Quality scoring modules
        self.scoring_modules = {
            'semantic_similarity': None,
            'tone_alignment': None,
            'structure_validator': None,
            'repetition_checker': None,
            'length_score': None
        }
        
        # Quality thresholds
        self.thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.75,
            'concerning': 0.65,
            'poor': 0.50
        }
        
        # Reviewer feedback options
        self.feedback_categories = {
            'semantic_issues': [
                'Meaning changed',
                'Logic errors',
                'Factual inaccuracies',
                'Context lost',
                'Semantic drift'
            ],
            'tone_issues': [
                'Wrong spiritual tone',
                'Inconsistent style',
                'Inappropriate language',
                'Tone mismatch',
                'Style deviation'
            ],
            'structure_issues': [
                'Format broken',
                'Q&A structure lost',
                'Poor organization',
                'Missing elements',
                'Structure unclear'
            ],
            'content_issues': [
                'Too repetitive',
                'Too verbose',
                'Too brief',
                'Filler content',
                'Low information density'
            ],
            'quality_issues': [
                'Hallucinations',
                'Unnecessary expansion',
                'Poor enhancement',
                'Quality degradation',
                'Enhancement failure'
            ]
        }
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for dashboard"""
        
        if 'quality_scores_history' not in st.session_state:
            st.session_state['quality_scores_history'] = []
        
        if 'reviewer_feedback_history' not in st.session_state:
            st.session_state['reviewer_feedback_history'] = []
        
        if 'quality_data_export' not in st.session_state:
            st.session_state['quality_data_export'] = []
        
        if 'dashboard_settings' not in st.session_state:
            st.session_state['dashboard_settings'] = {
                'auto_refresh': True,
                'show_details': True,
                'export_format': 'json'
            }
    
    def calculate_comprehensive_quality_score(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score using all modules
        
        Args:
            content_data: Dictionary containing content and analysis results
        
        Returns:
            Comprehensive quality scoring results
        """
        
        try:
            # Extract content information
            original_content = content_data.get('original_content', '')
            enhanced_content = content_data.get('enhanced_content', '')
            content_type = content_data.get('content_type', 'general')
            tone_style = content_data.get('tone_style', 'universal_wisdom')
            
            # Initialize results
            quality_results = {
                'timestamp': datetime.now().isoformat(),
                'content_type': content_type,
                'tone_style': tone_style,
                'module_scores': {},
                'overall_score': 0.0,
                'quality_level': 'unknown',
                'pass_status': False,
                'confidence': 0.0,
                'recommendations': [],
                'detailed_analysis': {}
            }
            
            # Calculate individual module scores
            module_scores = {}
            
            # 1. Semantic Similarity Score
            if 'semantic_analysis' in content_data:
                semantic_score = content_data['semantic_analysis'].get('similarity_score', 0.5)
                module_scores['semantic_similarity'] = semantic_score
                quality_results['detailed_analysis']['semantic'] = content_data['semantic_analysis']
            else:
                module_scores['semantic_similarity'] = 0.5
            
            # 2. Tone Alignment Score
            if 'tone_analysis' in content_data:
                tone_score = content_data['tone_analysis'].get('alignment_score', 0.5)
                module_scores['tone_alignment'] = tone_score
                quality_results['detailed_analysis']['tone'] = content_data['tone_analysis']
            else:
                module_scores['tone_alignment'] = 0.5
            
            # 3. Structure Validation Score
            if 'structure_analysis' in content_data:
                structure_score = content_data['structure_analysis'].get('validation_score', 0.5)
                module_scores['structure_validator'] = structure_score
                quality_results['detailed_analysis']['structure'] = content_data['structure_analysis']
            else:
                module_scores['structure_validator'] = 0.5
            
            # 4. Repetition Check Score
            if 'repetition_analysis' in content_data:
                repetition_score = content_data['repetition_analysis'].get('uniqueness_score', 0.5)
                module_scores['repetition_checker'] = repetition_score
                quality_results['detailed_analysis']['repetition'] = content_data['repetition_analysis']
            else:
                module_scores['repetition_checker'] = 0.5
            
            # 5. Length Score
            if 'length_analysis' in content_data:
                length_score = content_data['length_analysis'].get('overall_length_score', 0.5)
                module_scores['length_score'] = length_score
                quality_results['detailed_analysis']['length'] = content_data['length_analysis']
            else:
                module_scores['length_score'] = 0.5
            
            # Store module scores
            quality_results['module_scores'] = module_scores
            
            # Calculate weighted overall score
            weights = {
                'semantic_similarity': 0.30,  # Most important - meaning preservation
                'tone_alignment': 0.25,       # Critical for style consistency
                'structure_validator': 0.20,  # Important for format compliance
                'repetition_checker': 0.15,   # Important for content quality
                'length_score': 0.10          # Important but less critical
            }
            
            overall_score = sum(score * weights[module] for module, score in module_scores.items())
            quality_results['overall_score'] = overall_score
            
            # Calculate confidence based on score consistency
            scores = list(module_scores.values())
            mean_score = np.mean(scores)
            variance = np.var(scores)
            confidence = max(0.0, 1.0 - (variance * 4))  # Lower variance = higher confidence
            quality_results['confidence'] = confidence
            
            # Determine quality level and pass status
            quality_level, pass_status = self.determine_quality_level(overall_score, confidence)
            quality_results['quality_level'] = quality_level
            quality_results['pass_status'] = pass_status
            
            # Generate recommendations
            recommendations = self.generate_quality_recommendations(module_scores, quality_results['detailed_analysis'])
            quality_results['recommendations'] = recommendations
            
            # Log quality assessment
            log_event("quality_assessed", {
                "overall_score": overall_score,
                "quality_level": quality_level,
                "pass_status": pass_status,
                "confidence": confidence
            }, "quality_dashboard")
            
            return quality_results
        
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {str(e)}")
            return self.get_fallback_quality_results()
    
    def determine_quality_level(self, overall_score: float, confidence: float) -> Tuple[str, bool]:
        """Determine quality level and pass status"""
        
        # Adjust thresholds based on confidence
        confidence_factor = max(0.8, confidence)  # Minimum confidence factor
        adjusted_thresholds = {
            level: threshold * confidence_factor 
            for level, threshold in self.thresholds.items()
        }
        
        if overall_score >= adjusted_thresholds['excellent']:
            return 'excellent', True
        elif overall_score >= adjusted_thresholds['good']:
            return 'good', True
        elif overall_score >= adjusted_thresholds['acceptable']:
            return 'acceptable', True
        elif overall_score >= adjusted_thresholds['concerning']:
            return 'concerning', False
        else:
            return 'poor', False
    
    def generate_quality_recommendations(self, module_scores: Dict[str, float], detailed_analysis: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        
        recommendations = []
        
        # Semantic similarity recommendations
        semantic_score = module_scores.get('semantic_similarity', 0.5)
        if semantic_score < 0.75:
            recommendations.append("🔍 Improve semantic similarity - content meaning has drifted")
            if semantic_score < 0.5:
                recommendations.append("⚠️ Critical semantic drift detected - major revision needed")
        
        # Tone alignment recommendations
        tone_score = module_scores.get('tone_alignment', 0.5)
        if tone_score < 0.75:
            recommendations.append("🎭 Improve tone alignment - style doesn't match selected spiritual school")
            if tone_score < 0.5:
                recommendations.append("⚠️ Significant tone mismatch - reconsider enhancement approach")
        
        # Structure validation recommendations
        structure_score = module_scores.get('structure_validator', 0.5)
        if structure_score < 0.75:
            recommendations.append("🏗️ Fix structure issues - format compliance problems detected")
            if structure_score < 0.5:
                recommendations.append("⚠️ Major structure problems - format may be broken")
        
        # Repetition checker recommendations
        repetition_score = module_scores.get('repetition_checker', 0.5)
        if repetition_score < 0.75:
            recommendations.append("🔄 Reduce repetition - content contains redundant phrases")
            if repetition_score < 0.5:
                recommendations.append("⚠️ Excessive repetition detected - significant editing needed")
        
        # Length score recommendations
        length_score = module_scores.get('length_score', 0.5)
        if length_score < 0.75:
            recommendations.append("📏 Optimize length - content length issues detected")
            if length_score < 0.5:
                recommendations.append("⚠️ Significant length problems - major adjustment needed")
        
        # Overall quality recommendations
        overall_score = sum(module_scores.values()) / len(module_scores)
        if overall_score < 0.6:
            recommendations.append("🚨 Overall quality is poor - consider complete re-enhancement")
        elif overall_score < 0.75:
            recommendations.append("⚡ Multiple quality issues detected - systematic review recommended")
        
        return recommendations
    
    def render_quality_dashboard(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render comprehensive quality scoring dashboard
        
        Args:
            content_data: Content and analysis data
        
        Returns:
            Quality assessment results
        """
        
        st.subheader("📊 Quality Scoring Dashboard")
        
        # Calculate quality scores
        with st.spinner("Calculating comprehensive quality scores..."):
            quality_results = self.calculate_comprehensive_quality_score(content_data)
        
        # Render dashboard components
        self.render_overall_quality_metrics(quality_results)
        self.render_module_scores(quality_results)
        self.render_quality_assessment(quality_results)
        self.render_quality_recommendations(quality_results)
        
        # Render reviewer feedback section
        reviewer_feedback = self.render_reviewer_feedback_section(quality_results)
        
        # Update quality results with feedback
        if reviewer_feedback:
            quality_results['reviewer_feedback'] = reviewer_feedback
        
        # Store results for export
        self.store_quality_data(quality_results, content_data)
        
        return quality_results
    
    def render_overall_quality_metrics(self, quality_results: Dict[str, Any]):
        """Render overall quality metrics"""
        
        st.markdown("**📊 Overall Quality Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_score = quality_results.get('overall_score', 0.5)
            quality_level = quality_results.get('quality_level', 'unknown')
            
            # Color coding based on quality level
            color_map = {
                'excellent': 'normal',
                'good': 'normal', 
                'acceptable': 'normal',
                'concerning': 'inverse',
                'poor': 'inverse',
                'unknown': 'off'
            }
            
            st.metric(
                "Overall Quality",
                f"{overall_score:.1f}/10" if overall_score <= 1.0 else f"{overall_score*10:.1f}/10",
                delta=f"{(overall_score - 0.8)*10:.1f}" if overall_score != 0.8 else None,
                delta_color=color_map.get(quality_level, 'off')
            )
        
        with col2:
            confidence = quality_results.get('confidence', 0.5)
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta=f"{(confidence - 0.8)*100:.1f}%" if confidence != 0.8 else None,
                delta_color='normal' if confidence >= 0.8 else 'inverse'
            )
        
        with col3:
            pass_status = quality_results.get('pass_status', False)
            status_text = "✅ PASS" if pass_status else "❌ FAIL"
            st.metric(
                "Status",
                status_text,
                delta=None
            )
        
        with col4:
            quality_level = quality_results.get('quality_level', 'unknown')
            level_colors = {
                'excellent': '🟢',
                'good': '🟡',
                'acceptable': '🟠',
                'concerning': '🔴',
                'poor': '⚫',
                'unknown': '⚪'
            }
            st.metric(
                "Quality Level",
                f"{level_colors.get(quality_level, '⚪')} {quality_level.title()}",
                delta=None
            )
    
    def render_module_scores(self, quality_results: Dict[str, Any]):
        """Render individual module scores"""
        
        st.markdown("**🔍 Module Scores:**")
        
        module_scores = quality_results.get('module_scores', {})
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            semantic_score = module_scores.get('semantic_similarity', 0.5)
            st.metric(
                "Semantic Similarity",
                f"{semantic_score:.2f}",
                delta=f"{semantic_score - 0.92:.2f}" if semantic_score != 0.92 else None,
                delta_color='normal' if semantic_score >= 0.75 else 'inverse'
            )
        
        with col2:
            tone_score = module_scores.get('tone_alignment', 0.5)
            st.metric(
                "Tone Alignment", 
                f"{tone_score:.2f}",
                delta=f"{tone_score - 0.87:.2f}" if tone_score != 0.87 else None,
                delta_color='normal' if tone_score >= 0.75 else 'inverse'
            )
        
        with col3:
            structure_score = module_scores.get('structure_validator', 0.5)
            format_status = "✅" if structure_score >= 0.8 else "❌"
            st.metric(
                "Format Valid",
                format_status,
                delta=None
            )
        
        with col4:
            repetition_score = module_scores.get('repetition_checker', 0.5)
            st.metric(
                "Uniqueness",
                f"{repetition_score:.2f}",
                delta=f"{repetition_score - 0.85:.2f}" if repetition_score != 0.85 else None,
                delta_color='normal' if repetition_score >= 0.75 else 'inverse'
            )
        
        with col5:
            length_score = module_scores.get('length_score', 0.5)
            st.metric(
                "Length Score",
                f"{length_score:.2f}",
                delta=f"{length_score - 0.75:.2f}" if length_score != 0.75 else None,
                delta_color='normal' if length_score >= 0.75 else 'inverse'
            )
        
        # Render score visualization
        self.render_score_visualization(module_scores)
    
    def render_score_visualization(self, module_scores: Dict[str, float]):
        """Render score visualization chart"""
        
        # Create radar chart
        categories = list(module_scores.keys())
        scores = list(module_scores.values())
        
        # Clean up category names for display
        display_names = {
            'semantic_similarity': 'Semantic\nSimilarity',
            'tone_alignment': 'Tone\nAlignment',
            'structure_validator': 'Structure\nValidation',
            'repetition_checker': 'Repetition\nCheck',
            'length_score': 'Length\nScore'
        }
        
        display_categories = [display_names.get(cat, cat) for cat in categories]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=display_categories,
            fill='toself',
            name='Quality Scores',
            line_color='rgb(0, 123, 255)',
            fillcolor='rgba(0, 123, 255, 0.3)'
        ))
        
        # Add threshold line
        threshold_scores = [0.75] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=threshold_scores,
            theta=display_categories,
            mode='lines',
            name='Acceptable Threshold',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Quality Scores Radar Chart",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_quality_assessment(self, quality_results: Dict[str, Any]):
        """Render quality assessment summary"""
        
        st.markdown("**📋 Quality Assessment:**")
        
        overall_score = quality_results.get('overall_score', 0.5)
        quality_level = quality_results.get('quality_level', 'unknown')
        pass_status = quality_results.get('pass_status', False)
        confidence = quality_results.get('confidence', 0.5)
        
        # Status message
        if pass_status:
            if quality_level == 'excellent':
                st.success(f"🎉 Excellent quality! Overall score: {overall_score:.3f}")
            elif quality_level == 'good':
                st.success(f"✅ Good quality! Overall score: {overall_score:.3f}")
            else:
                st.success(f"✅ Acceptable quality. Overall score: {overall_score:.3f}")
        else:
            if quality_level == 'concerning':
                st.warning(f"⚠️ Quality concerns detected. Overall score: {overall_score:.3f}")
            else:
                st.error(f"❌ Poor quality detected. Overall score: {overall_score:.3f}")
        
        # Confidence indicator
        if confidence >= 0.8:
            st.info(f"🎯 High confidence assessment ({confidence:.1%})")
        elif confidence >= 0.6:
            st.info(f"🎯 Moderate confidence assessment ({confidence:.1%})")
        else:
            st.warning(f"⚠️ Low confidence assessment ({confidence:.1%}) - manual review recommended")
    
    def render_quality_recommendations(self, quality_results: Dict[str, Any]):
        """Render quality improvement recommendations"""
        
        recommendations = quality_results.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Quality Improvement Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("🎉 No specific recommendations - quality analysis looks excellent!")
    
    def render_reviewer_feedback_section(self, quality_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Render reviewer feedback section"""
        
        st.markdown("**👥 Reviewer Feedback:**")
        
        # Quick feedback buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍 Approve", key="approve_btn"):
                feedback = {
                    'action': 'approve',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': quality_results.get('overall_score', 0.5)
                }
                self.store_reviewer_feedback(feedback)
                st.success("✅ Content approved!")
                return feedback
        
        with col2:
            if st.button("📝 Mark for Rework", key="rework_btn"):
                feedback = {
                    'action': 'mark_for_rework',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': quality_results.get('overall_score', 0.5)
                }
                self.store_reviewer_feedback(feedback)
                st.warning("📝 Marked for rework!")
                return feedback
        
        with col3:
            if st.button("❌ Reject", key="reject_btn"):
                feedback = {
                    'action': 'reject',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': quality_results.get('overall_score', 0.5)
                }
                self.store_reviewer_feedback(feedback)
                st.error("❌ Content rejected!")
                return feedback
        
        with col4:
            if st.button("🔄 Re-analyze", key="reanalyze_btn"):
                st.rerun()
        
        # Detailed feedback form
        with st.expander("📝 Detailed Feedback", expanded=False):
            feedback_data = {}
            
            # Issue categories
            st.markdown("**Select Issues (if any):**")
            
            for category, issues in self.feedback_categories.items():
                selected_issues = st.multiselect(
                    f"{category.replace('_', ' ').title()}:",
                    issues,
                    key=f"issues_{category}"
                )
                if selected_issues:
                    feedback_data[category] = selected_issues
            
            # Free text feedback
            feedback_text = st.text_area(
                "Additional Comments:",
                placeholder="Provide specific feedback about quality issues or improvements...",
                key="feedback_text"
            )
            
            # Quality rating
            quality_rating = st.slider(
                "Manual Quality Rating (1-10):",
                min_value=1,
                max_value=10,
                value=int(quality_results.get('overall_score', 0.5) * 10),
                key="quality_rating"
            )
            
            # Submit detailed feedback
            if st.button("Submit Detailed Feedback", key="submit_feedback"):
                detailed_feedback = {
                    'action': 'detailed_feedback',
                    'timestamp': datetime.now().isoformat(),
                    'issues': feedback_data,
                    'comments': feedback_text,
                    'manual_rating': quality_rating / 10.0,
                    'auto_score': quality_results.get('overall_score', 0.5)
                }
                self.store_reviewer_feedback(detailed_feedback)
                st.success("📝 Detailed feedback submitted!")
                return detailed_feedback
        
        return None
    
    def store_reviewer_feedback(self, feedback: Dict[str, Any]):
        """Store reviewer feedback in session state"""
        
        st.session_state['reviewer_feedback_history'].append(feedback)
        
        # Log feedback
        log_event("reviewer_feedback", {
            "action": feedback.get('action', 'unknown'),
            "auto_score": feedback.get('auto_score', 0.5),
            "manual_rating": feedback.get('manual_rating')
        }, "quality_dashboard")
    
    def store_quality_data(self, quality_results: Dict[str, Any], content_data: Dict[str, Any]):
        """Store quality data for export and analysis"""
        
        # Create export record
        export_record = {
            "input": content_data.get('original_content', ''),
            "output": content_data.get('enhanced_content', ''),
            "scores": {
                "semantic_similarity": quality_results['module_scores'].get('semantic_similarity', 0.5),
                "tone_alignment": quality_results['module_scores'].get('tone_alignment', 0.5),
                "structure_validation": quality_results['module_scores'].get('structure_validator', 0.5),
                "repetition_check": quality_results['module_scores'].get('repetition_checker', 0.5),
                "length_score": quality_results['module_scores'].get('length_score', 0.5),
                "overall_score": quality_results.get('overall_score', 0.5)
            },
            "final_label": "pass" if quality_results.get('pass_status', False) else "fail",
            "quality_level": quality_results.get('quality_level', 'unknown'),
            "confidence": quality_results.get('confidence', 0.5),
            "recommendations": quality_results.get('recommendations', []),
            "timestamp": quality_results.get('timestamp', datetime.now().isoformat()),
            "content_type": quality_results.get('content_type', 'general'),
            "tone_style": quality_results.get('tone_style', 'universal_wisdom')
        }
        
        # Add reviewer feedback if available
        if 'reviewer_feedback' in quality_results:
            export_record['reviewer_notes'] = quality_results['reviewer_feedback']
        
        # Store in session state
        st.session_state['quality_data_export'].append(export_record)
        st.session_state['quality_scores_history'].append(quality_results)
    
    def render_quality_history(self):
        """Render quality scoring history and analytics"""
        
        if not st.session_state['quality_scores_history']:
            st.info("No quality history available yet.")
            return
        
        st.subheader("📈 Quality History & Analytics")
        
        # Convert history to DataFrame
        history_data = []
        for record in st.session_state['quality_scores_history']:
            history_data.append({
                'timestamp': record.get('timestamp', ''),
                'overall_score': record.get('overall_score', 0.5),
                'quality_level': record.get('quality_level', 'unknown'),
                'pass_status': record.get('pass_status', False),
                'confidence': record.get('confidence', 0.5)
            })
        
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Quality trend chart
        fig = px.line(
            df, 
            x='timestamp', 
            y='overall_score',
            title='Quality Score Trend',
            labels={'overall_score': 'Overall Quality Score', 'timestamp': 'Time'}
        )
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Acceptable Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality distribution
        col1, col2 = st.columns(2)
        
        with col1:
            quality_counts = df['quality_level'].value_counts()
            fig_pie = px.pie(
                values=quality_counts.values,
                names=quality_counts.index,
                title='Quality Level Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            pass_counts = df['pass_status'].value_counts()
            fig_bar = px.bar(
                x=['Pass', 'Fail'],
                y=[pass_counts.get(True, 0), pass_counts.get(False, 0)],
                title='Pass/Fail Distribution',
                color=['Pass', 'Fail'],
                color_discrete_map={'Pass': 'green', 'Fail': 'red'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def export_quality_data(self, format_type: str = 'json') -> str:
        """Export quality data for fine-tuning"""
        
        export_data = st.session_state['quality_data_export']
        
        if format_type == 'json':
            return json.dumps(export_data, indent=2)
        elif format_type == 'jsonl':
            return '\n'.join(json.dumps(record) for record in export_data)
        elif format_type == 'csv':
            df = pd.DataFrame(export_data)
            return df.to_csv(index=False)
        else:
            return json.dumps(export_data, indent=2)
    
    def get_fallback_quality_results(self) -> Dict[str, Any]:
        """Return fallback results when quality calculation fails"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'content_type': 'unknown',
            'tone_style': 'unknown',
            'module_scores': {
                'semantic_similarity': 0.5,
                'tone_alignment': 0.5,
                'structure_validator': 0.5,
                'repetition_checker': 0.5,
                'length_score': 0.5
            },
            'overall_score': 0.5,
            'quality_level': 'unknown',
            'pass_status': False,
            'confidence': 0.3,
            'recommendations': ['Quality analysis failed - manual review required'],
            'detailed_analysis': {}
        }

# Integration functions for main app
def create_quality_dashboard() -> QualityScoringDashboard:
    """Create quality scoring dashboard instance"""
    return QualityScoringDashboard()

def assess_content_quality(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess content quality using comprehensive scoring
    
    Usage:
    from modules.quality_scoring_dashboard import assess_content_quality
    
    quality_results = assess_content_quality({
        'original_content': original_text,
        'enhanced_content': enhanced_text,
        'content_type': 'qa_format',
        'tone_style': 'zen_buddhism',
        'semantic_analysis': semantic_results,
        'tone_analysis': tone_results,
        'structure_analysis': structure_results,
        'repetition_analysis': repetition_results,
        'length_analysis': length_results
    })
    """
    
    dashboard = QualityScoringDashboard()
    return dashboard.calculate_comprehensive_quality_score(content_data)

if __name__ == "__main__":
    # Test the quality dashboard
    st.set_page_config(page_title="Quality Dashboard Test", layout="wide")
    
    st.title("Quality Scoring Dashboard Test")
    
    # Sample content data
    sample_data = {
        'original_content': "What is meditation?",
        'enhanced_content': "What is meditation? Meditation is a practice of focused attention and mindfulness that cultivates inner peace and awareness.",
        'content_type': 'qa_format',
        'tone_style': 'zen_buddhism',
        'semantic_analysis': {'similarity_score': 0.92},
        'tone_analysis': {'alignment_score': 0.87},
        'structure_analysis': {'validation_score': 0.95},
        'repetition_analysis': {'uniqueness_score': 0.88},
        'length_analysis': {'overall_length_score': 0.75}
    }
    
    if st.button("Test Quality Dashboard"):
        dashboard = QualityScoringDashboard()
        results = dashboard.render_quality_dashboard(sample_data)
        
        st.json(results)

