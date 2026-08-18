#include<iostream>
using namespace std;
class node{
    public:
    int data;
    node* next;
    node(int data){
        this->data=data;
        next=NULL;
    }
};
void insertathead(node* &head, int data){
    node* temp= new node(data);
    temp->next=head;
    head=temp;
}
void inserttail(node* &head, int data){
    node* temp1= new node(data);
    node* temp=head;
    while(temp->next!=nullptr){
        
        temp=temp->next;
    }
    temp->next=temp1;
}

int main(){
    node* head= new node(34);
    head->next=new node(87);
    insertathead(head,78);
    inserttail(head,9000);
    node* temp=head;

    while(temp!=nullptr){
        cout<<temp->data<<" ";
        temp=temp->next;
    }

}