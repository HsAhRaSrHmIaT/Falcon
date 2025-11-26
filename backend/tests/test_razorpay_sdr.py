#!/usr/bin/env python3
"""
Test script for the Razorpay SDR Agent
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import the agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import SDRData, LeadData #type: ignore

async def test_razorpay_data_loading():
    """Test loading Razorpay company data"""
    print("Testing Razorpay Company Data Loading...")
    
    company_data_file = Path(__file__).parent / "shared-data" / "razorpay_company_data.json"
    
    try:
        if company_data_file.exists():
            with open(company_data_file, 'r', encoding='utf-8') as f:
                company_data = json.load(f)
                
            print(f"✅ Company: {company_data['company']['name']}")
            print(f"✅ Tagline: {company_data['company']['tagline']}")
            print(f"✅ Description: {company_data['company']['description'][:100]}...")
            print(f"✅ Number of FAQ items: {len(company_data['faq'])}")
            print(f"✅ Number of products: {len(company_data['products'])}")
            
            # Test FAQ content
            print("\\n📋 Sample FAQ items:")
            for i, faq in enumerate(company_data['faq'][:3]):
                print(f"{i+1}. Q: {faq['question']}")
                print(f"   A: {faq['answer'][:80]}...")
                
            # Test Products
            print("\\n💼 Products offered:")
            for product_key, product in company_data['products'].items():
                print(f"- {product['name']}: {product['description']}")
                
            return True
        else:
            print(f"❌ Company data file not found: {company_data_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading company data: {e}")
        return False

def test_lead_data_structure():
    """Test lead data structure"""
    print("\\nTesting Lead Data Structure...")
    
    # Test LeadData
    lead = LeadData()
    print(f"✅ Empty lead missing fields: {lead.get_missing_fields()}")
    
    # Simulate Razorpay-specific lead
    lead.name = "Arjun Sharma"
    lead.company = "TechFlow Solutions"
    lead.email = "arjun@techflow.in"
    lead.role = "CTO"
    lead.use_case = "Need payment gateway for e-commerce platform"
    lead.team_size = "25 employees"
    lead.timeline = "Next month"
    
    print(f"✅ Complete lead missing fields: {lead.get_missing_fields()}")
    print(f"✅ Lead completion status: {lead.get_completion_status()}")
    
    return True

def create_sample_razorpay_lead():
    """Create a sample Razorpay lead file"""
    print("\\nCreating Sample Razorpay Lead...")
    
    leads_dir = Path(__file__).parent / "leads"
    leads_dir.mkdir(exist_ok=True)
    
    sample_lead = {
        "name": "Vikram Patel",
        "company": "Digital Commerce India",
        "email": "vikram@digitalcommerce.in",
        "role": "Founder & CEO",
        "use_case": "Need comprehensive payment solution for our marketplace platform with international payment support",
        "team_size": "50-100 employees",
        "timeline": "Within next 6 weeks",
        "notes": "Currently using multiple payment providers, wants to consolidate",
        "call_start_time": datetime.now().isoformat(),
        "call_end_time": datetime.now().isoformat(),
        "call_summary": "Spoke with Vikram Patel who is a Founder & CEO at Digital Commerce India. They're interested in: Need comprehensive payment solution for our marketplace platform. Timeline: Within next 6 weeks. Team size: 50-100 employees."
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = leads_dir / f"sample_lead_razorpay_{timestamp}.json"
    
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_lead, f, indent=2)
        
        print(f"✅ Sample lead file created: {sample_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample lead file: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 === Razorpay SDR Agent Test Suite ===\\n")
    
    test_results = []
    
    # Test company data loading
    result1 = await test_razorpay_data_loading()
    test_results.append(("Razorpay Data Loading", result1))
    
    # Test lead data structures
    result2 = test_lead_data_structure()
    test_results.append(("Lead Data Structure", result2))
    
    # Create sample lead file
    result3 = create_sample_razorpay_lead()
    test_results.append(("Sample Lead Creation", result3))
    
    # Print results
    print("\\n=== Test Results ===")
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    print(f"\\nOverall: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}")
    
    if all_passed:
        print("\\n🎉 Razorpay SDR Agent is ready to use!")
        print("\\n📖 To use the Razorpay SDR agent:")
        print("1. Start your LiveKit server")
        print("2. Create a room (any name works)")
        print("3. The agent will act as Priya, an SDR for Razorpay")
        print("4. Lead data will be saved with 'lead_razorpay_' prefix")
        print("5. Agent can answer questions about:")
        print("   - Payment gateway solutions")
        print("   - Business banking (RazorpayX)")
        print("   - Business loans (Capital)")
        print("   - Payroll management")
        print("6. Agent will collect business information naturally")
    else:
        print("\\n⚠️  Please fix the failed tests before using the agent")

if __name__ == "__main__":
    asyncio.run(main())