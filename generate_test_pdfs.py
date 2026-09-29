from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_50_page_pdf(filename, department_title, policy_focus):
    print(f"📄 Building 50-page document: {filename}...")
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=15,
        textColor=colors.navy
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=10,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )

    story = []

    # Explicitly loop 50 times to force 50 pages with page breaks
    for page in range(1, 51):
        story.append(Paragraph(f"{department_title} - Master Policy Manual", title_style))
        story.append(Paragraph(f"Section {page}: Guidelines on {policy_focus} (Page {page} of 50)", heading_style))
        
        story.append(Paragraph(
            f"<b>Clause {page}.1 Overview:</b> This official internal document governs all standard operating procedures, compliance boundaries, and protocol enforcement associated with {policy_focus.lower()} across enterprise branches. All personnel are strictly required to adhere to the provisions outlined in this section.",
            body_style
        ))
        
        story.append(Paragraph(
            f"<b>Clause {page}.2 Compliance & Security:</b> Unauthorized disclosure, circumvention, or negligence regarding {policy_focus.lower()} protocols will result in immediate internal audit review, compliance investigation, and potential disciplinary measures in accordance with internal governance standards.",
            body_style
        ))
        
        story.append(Paragraph(
            f"<b>Clause {page}.3 Operational Workflow:</b> Sub-sections under page {page} mandate that any exception to standard practices must be formally logged, cross-verified by department supervisors, and archived within the secured local repository for a mandatory compliance retention period of seven years.",
            body_style
        ))
        
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"<i>Confidential & Proprietary - Internal Banking Operations Only [Page {page}]</i>", styles['Italic']))
        
        # Force a physical page break on every iteration except the very last page
        if page < 50:
            story.append(PageBreak())

    doc.build(story)
    print(f"✅ Successfully created {filename} (50 pages).")

if __name__ == "__main__":
    create_50_page_pdf("hr_policy_50pages.pdf", "Human Resources", "Employee Conduct, Benefits, and Leave Management")
    create_50_page_pdf("finance_policy_50pages.pdf", "Finance Department", "Corporate Budgets, Expense Audits, and Financial Limits")
    create_50_page_pdf("operations_policy_50pages.pdf", "Operations Department", "Physical Bank Vaults and Branch Security Protocols")
    create_50_page_pdf("general_policy_50pages.pdf", "General Operations", "Enterprise Workplace Code of Ethics and Public Safety")
    print("\n🎉 All four 50-page master policy PDFs generated successfully!")